"""The control loop: gather signals, ask :mod:`plan`, execute, repeat.

Deliberately dull. Every judgement lives in :func:`plan.decide`; this only does I/O
and error classification.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import diagnostics, dispatch, fleet, signals
from .plan import FleetState, Limits, decide

logger = logging.getLogger(__name__)


@dataclass
class Config:
    template: Path
    manager_ip: str
    master_key_hex: str
    redis_password: str
    #: Which deployment this controller owns, tagged onto every instance it creates and
    #: required of every instance it will touch. See fleet.DEPLOYMENT_TAG_PREFIX: the
    #: test mirror and production share one OpenStack project, so without this each
    #: controller counts and reaps the other's workers.
    deployment: str
    dispatch_queue: str = "sivacor"
    girder_host: str | None = None
    worker_image: str | None = None
    #: What a worker VM's celery subscribes to. ``None`` leaves the template's default,
    #: ``sivacor,<its private queue>``. Set it to the private queue alone once this
    #: deployment has armed targeted assignment and nothing publishes to the shared
    #: queue -- and only then: a worker that has stopped consuming ``sivacor`` cannot
    #: serve a submission dispatched the old way, so this is the step that ends the
    #: one-flag rollback (P2 rollout step 4).
    worker_queues: str | None = None
    image: str = "Featured-Ubuntu24"
    flavor: str = "m3.medium"
    network: str = "auto_allocated_network"
    key_name: str | None = None
    security_groups: list[str] = field(default_factory=list)
    interval: float = 30.0
    #: Where to write a post-mortem before deleting an instance that died badly.
    #: ``None`` disables capture entirely -- the pre-2026-08-11 behaviour, where a
    #: worker that powered off mid-run was deleted with no record of why. Point it at
    #: a bind mount: a path inside the container dies with the container.
    diagnostics_dir: Path | None = None
    #: How long the breaker stays open before ONE creation attempt is allowed again.
    #: Non-negotiable that this exists at all: see Controller._expire_breaker.
    breaker_cooldown: float = 900.0
    limits: Limits = field(default_factory=Limits)


class Controller:
    def __init__(self, conn, redis_client, db, cfg: Config):
        self.conn = conn
        self.redis = redis_client
        #: Girder's MongoDB database. Read-only here; see signals.JOB_COLLECTION for
        #: why this is the database rather than the REST API.
        self.db = db
        self.cfg = cfg
        #: Consecutive failed creation attempts. Cleared by a successful create, and
        #: by the cooldown in :meth:`_expire_breaker` -- which is what makes the
        #: breaker recoverable at all. The comment here used to claim it was "reset by
        #: any successful round"; it was not reset anywhere, and that cost a production
        #: outage on 2026-08-12.
        self.consecutive_failures = 0
        #: ``time.monotonic()`` of the last failure, or None. Monotonic on purpose: a
        #: clock step must not extend or collapse the cooldown.
        self._last_failure: float | None = None
        #: Instances observed to have claimed a submission, remembered for as long as
        #: OpenStack still reports them. See :meth:`_spent`.
        self._spent_seen: set[str] = set()
        #: Last observed value of Girder's arm flag, so a change can be logged once
        #: rather than every tick. Arming is the highest-stakes event this process
        #: takes part in and it happens with no restart and no deploy, so it must
        #: leave a line in the log to correlate an incident against.
        self._armed: bool | None = None

    def _spent(self, instances) -> frozenset[str]:
        """Instance ids that have claimed a submission, monotonically.

        ``signals.spent_instance_ids`` derives the set from surviving Girder job
        documents, and **a user can delete those**: ``DELETE /sivacor/submission/:id``
        calls ``Job().remove()``, which erases the ``meta.worker_queue`` marker the
        signal is made of. Deletion is only allowed once a submission is *completed* --
        but an ephemeral worker lives on for the boot grace plus its idle clock, ~15
        min, and the submission is deletable for all of it.

        Observed in production 2026-08-05: a submission was deleted at 15:06:54, its
        instance dropped out of ``spent`` on the same tick, and the fleet then reported
        ``1 available of 1 live`` for 12 min 36 s -- for a worker that had already
        cancelled its dispatch-queue consumer at claim time (P3.2) and was counting
        down to poweroff. A submission arriving in that window computes
        ``depth - available == 0``, gets no instance, and nothing consumes it: the run
        3 / run 5 stall class, reached through a path no loop test exercised.

        A claim is irreversible, so remembering it is sound: an instance that has taken
        a submission will never take another. Retention is bounded by the fleet itself
        -- ids OpenStack no longer reports are dropped, which also discards the
        ``sivacor.static-01`` style entries the signal yields for the manager's own
        static worker, since those never match an instance id.

        **The memory is per process.** A controller restart while a spent worker is
        still winding down re-opens the same window until that instance is reaped. The
        durable fix is a claim marker in Redis alongside D9's readiness marker, which
        no Girder deletion can touch -- but that is a ``girder-sivacor`` change and so
        a worker-image change, and the fleet currently rides a tag CI no longer builds.
        """
        known = {i.id for i in instances}
        self._spent_seen |= signals.spent_instance_ids(self.db, self.cfg.dispatch_queue)
        self._spent_seen &= known
        return frozenset(self._spent_seen)

    def limits(self) -> Limits:
        """This tick's limits, with ``assign`` taken from Girder's setting.

        The arm flag is not configuration of this process: it is one value both
        ``submit_job`` and this controller read, so that no state exists in which one
        is flipped and the other is not. See :func:`signals.targeted_assignment`. It
        is read once per tick and threaded through ``gather`` and ``decide`` together,
        so a flip mid-round cannot make those two disagree either.
        """
        armed = signals.targeted_assignment(self.db)
        if armed != self._armed:
            # The first reading is a statement of state, not a transition. "now OFF ...
            # no longer" at startup reads as though someone had just disarmed it, which
            # is precisely wrong for the line an operator greps to confirm what a
            # freshly deployed controller is doing -- the first use this line ever had
            # (mirror, 2026-08-19).
            logger.warning(
                "targeted assignment %s %s (%s): %s",
                "is" if self._armed is None else "is now",
                "ON" if armed else "OFF",
                signals.TARGETED_ASSIGNMENT_KEY,
                (
                    "this controller places submissions"
                    if armed
                    else "Girder dispatches them to the shared queue"
                ),
            )
            self._armed = armed
        return replace(self.cfg.limits, assign=armed)

    def gather(self, limits: Limits | None = None) -> FleetState:
        limits = self.cfg.limits if limits is None else limits
        # Instances first: _spent() prunes against them, and an id OpenStack no longer
        # reports must not linger in the cache.
        instances = fleet.list_fleet(self.conn, self.cfg.deployment)
        return FleetState(
            queue_depth=signals.queue_depth(self.redis, self.cfg.dispatch_queue),
            serving=signals.serving_count(self.db),
            spent=self._spent(instances),
            # Demand queue_depth cannot see -- a message a worker reserved but cannot
            # start has left the Redis list while being just as unserved -- and, under
            # Limits.assign, the assigner's work list too: one query, so capacity and
            # placement cannot be decided from two readings a tick apart. Swallows its
            # own errors: () under-provisions for one round and assigns nothing, whereas
            # raising would skip the round, and a round that does not happen is a round
            # that does not reap.
            waiting=signals.waiting_submissions(self.db),
            # Diagnostics only, and swallows its own errors for the same reason.
            running_jobs=signals.running_jobs_by_instance(
                self.db, self.cfg.dispatch_queue
            ),
            # Only read when something will consult it: gather() is all-or-nothing, so
            # an unused signal must not be able to fail a round. Assignment makes it
            # mandatory, which is easy to miss because the deadline check is off in
            # production -- an unregistered instance is never assigned to, so an empty
            # set here assigns nothing, forever, while every other number reads healthy.
            ready=(
                signals.ready_instance_ids(self.redis)
                if limits.provision_deadline is not None or limits.assign
                else frozenset()
            ),
            instances=instances,
            consecutive_failures=self.consecutive_failures,
        )

    def _expire_breaker(self) -> None:
        """Reopen the breaker after a quiet period, because nothing else can.

        ``plan.decide`` forces ``create = 0`` while the counter sits at the threshold,
        so a tripped breaker prevents the very success that would clear it: resetting
        on success is necessary and **not sufficient**. Without a time-based reopen it
        latches until the process is restarted by hand.

        Measured 2026-08-12: an oversized ``user_data`` (65 bytes over Nova's limit)
        tripped it in about 90 s, and it then refused every create for five minutes --
        through the fix being deployed -- logging only the ordinary "BREAKER OPEN"
        line. The fleet was down until someone restarted the service.

        One attempt, not a reset to trusting: if that attempt fails the counter is back
        at the threshold immediately and the fleet waits another cooldown. That keeps
        the property the breaker exists for -- a genuinely broken image cannot loop
        burning SUs -- while making an outage self-limiting rather than permanent.
        """
        if not self.consecutive_failures or self._last_failure is None:
            return
        if time.monotonic() - self._last_failure < self.cfg.breaker_cooldown:
            return
        logger.warning(
            "breaker: %.0fs since the last creation failure, allowing one attempt "
            "again (was %d consecutive)",
            self.cfg.breaker_cooldown,
            self.consecutive_failures,
        )
        self.consecutive_failures = 0
        self._last_failure = None

    def step(self) -> None:
        """One iteration. Never raises for an expected condition."""
        self._expire_breaker()
        try:
            # The arm flag first: gather() reads a signal it would otherwise skip, and
            # both halves of this tick have to be decided from one reading of it.
            limits = self.limits()
            state = self.gather(limits)
        except Exception:
            # Deciding on partial information is how a controller creates instances it
            # does not need. Skip the round; the next one is 30 s away.
            logger.warning("skipping round: could not gather state", exc_info=True)
            return

        decision = decide(state, limits)
        # alerts is a subset of reasons, so filter rather than log the anomalies twice.
        alerts = set(decision.alerts)
        for reason in decision.reasons:
            if reason not in alerts:
                logger.info("%s", reason)
        for alert in decision.alerts:
            logger.warning("%s", alert)

        by_id = {i.id: i for i in state.instances}

        # Deletes first: they free slots the creates may want, and they must happen
        # even when the breaker has blocked creation.
        for instance_id in decision.delete:
            # Strictly before the delete: Nova drops the console buffer with the
            # server, so this ordering is the entire value of the capture.
            if instance_id in decision.abnormal and self.cfg.diagnostics_dir:
                diagnostics.capture(
                    self.conn,
                    self.cfg.diagnostics_dir,
                    by_id[instance_id],
                    why=decision.reap_reasons.get(instance_id, "unrecorded"),
                    job=state.running_jobs.get(instance_id),
                )
            try:
                fleet.delete_instance(self.conn, instance_id)
            except Exception:
                logger.warning("could not delete %s", instance_id, exc_info=True)

        # Assignments before creates: placing work on an instance that already exists is
        # the cheap half, and a create that fails must not stop it. Each binding is
        # independent -- one submission whose chain will not build must not strand the
        # rest of the round -- and none of them touch the breaker, which counts *instance
        # creation* failures and would stop the whole fleet if a bad submission could
        # feed it.
        for submission_id, instance_id in decision.assign:
            try:
                dispatch.assign(
                    self.db, submission_id, instance_id, self.cfg.dispatch_queue
                )
            except Exception:
                logger.warning(
                    "could not assign submission %s to %s",
                    submission_id,
                    instance_id,
                    exc_info=True,
                )

        for rung in decision.create:
            try:
                # `rung` is the catalogue size this instance must be, or None for
                # "unsized" -- demand read from queue depth, which carries no size.
                #
                # **Every rung is None until P3.2 wires the catalogue**, because
                # `decide()` only emits sized creates when `Limits.sizes` is non-empty
                # and nothing populates it yet. So the flavour below is still the one
                # `SIVACOR_OS_FLAVOR` names, exactly as before, and this loop is a
                # rename of `range(decision.create)`. Asserting it rather than
                # commenting it: a sized rung arriving here before the mapping exists
                # would silently boot the wrong shape.
                if rung is not None:
                    logger.warning(
                        "decision asked for a %s GB instance but this build has no "
                        "size->flavor mapping yet (P3.2); booting %s regardless",
                        rung,
                        self.cfg.flavor,
                    )
                user_data = fleet.build_user_data(
                    self.cfg.template,
                    master_key_hex=self.cfg.master_key_hex,
                    redis_password=self.cfg.redis_password,
                    manager_ip=self.cfg.manager_ip,
                    girder_host=self.cfg.girder_host,
                    worker_image=self.cfg.worker_image,
                    worker_queues=self.cfg.worker_queues,
                )
                fleet.create_instance(self.conn, self.cfg, user_data)
                # A create that works is the only positive evidence that whatever
                # tripped the breaker is over. Nothing else cleared this counter
                # before 2026-08-12.
                self.consecutive_failures = 0
                self._last_failure = None
            except fleet.QuotaExceeded as exc:
                # Backpressure, not failure: the submission stays queued and this must
                # not count towards the breaker, or a full allocation would stop the
                # fleet scaling precisely when it is busiest.
                logger.info("allocation full, leaving work queued: %s", exc)
                break
            except Exception:
                self.consecutive_failures += 1
                self._last_failure = time.monotonic()
                logger.warning(
                    "instance creation failed (%d consecutive)",
                    self.consecutive_failures,
                    exc_info=True,
                )
                break

    def run(self) -> None:
        logger.info(
            "controller starting: deployment=%s queue=%s cap=%d interval=%.0fs",
            self.cfg.deployment,
            self.cfg.dispatch_queue,
            self.cfg.limits.max_instances,
            self.cfg.interval,
        )
        while True:
            self.step()
            time.sleep(self.cfg.interval)
