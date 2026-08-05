"""The control loop: gather signals, ask :mod:`plan`, execute, repeat.

Deliberately dull. Every judgement lives in :func:`plan.decide`; this only does I/O
and error classification.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import fleet, signals
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
    image: str = "Featured-Ubuntu24"
    flavor: str = "m3.medium"
    network: str = "auto_allocated_network"
    key_name: str | None = None
    security_groups: list[str] = field(default_factory=list)
    interval: float = 30.0
    limits: Limits = field(default_factory=Limits)


class Controller:
    def __init__(self, conn, redis_client, db, cfg: Config):
        self.conn = conn
        self.redis = redis_client
        #: Girder's MongoDB database. Read-only here; see signals.JOB_COLLECTION for
        #: why this is the database rather than the REST API.
        self.db = db
        self.cfg = cfg
        #: Consecutive instances that booted but never registered with celery. Reset
        #: by any successful round, so a transient failure does not accumulate towards
        #: the breaker.
        self.consecutive_failures = 0
        #: Instances observed to have claimed a submission, remembered for as long as
        #: OpenStack still reports them. See :meth:`_spent`.
        self._spent_seen: set[str] = set()

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

    def gather(self) -> FleetState:
        # Instances first: _spent() prunes against them, and an id OpenStack no longer
        # reports must not linger in the cache.
        instances = fleet.list_fleet(self.conn, self.cfg.deployment)
        return FleetState(
            queue_depth=signals.queue_depth(self.redis, self.cfg.dispatch_queue),
            serving=signals.serving_count(self.db),
            spent=self._spent(instances),
            # Only read when the deadline check is armed. Skipping the call when it is
            # disabled keeps a Redis hiccup from failing rounds for a signal nothing
            # would have consulted -- gather() is all-or-nothing by design.
            ready=(
                signals.ready_instance_ids(self.redis)
                if self.cfg.limits.provision_deadline is not None
                else frozenset()
            ),
            instances=instances,
            consecutive_failures=self.consecutive_failures,
        )

    def step(self) -> None:
        """One iteration. Never raises for an expected condition."""
        try:
            state = self.gather()
        except Exception:
            # Deciding on partial information is how a controller creates instances it
            # does not need. Skip the round; the next one is 30 s away.
            logger.warning("skipping round: could not gather state", exc_info=True)
            return

        decision = decide(state, self.cfg.limits)
        for reason in decision.reasons:
            logger.info("%s", reason)

        # Deletes first: they free slots the creates may want, and they must happen
        # even when the breaker has blocked creation.
        for instance_id in decision.delete:
            try:
                fleet.delete_instance(self.conn, instance_id)
            except Exception:
                logger.warning("could not delete %s", instance_id, exc_info=True)

        for _ in range(decision.create):
            try:
                user_data = fleet.build_user_data(
                    self.cfg.template,
                    master_key_hex=self.cfg.master_key_hex,
                    redis_password=self.cfg.redis_password,
                    manager_ip=self.cfg.manager_ip,
                    girder_host=self.cfg.girder_host,
                    worker_image=self.cfg.worker_image,
                )
                fleet.create_instance(self.conn, self.cfg, user_data)
            except fleet.QuotaExceeded as exc:
                # Backpressure, not failure: the submission stays queued and this must
                # not count towards the breaker, or a full allocation would stop the
                # fleet scaling precisely when it is busiest.
                logger.info("allocation full, leaving work queued: %s", exc)
                break
            except Exception:
                self.consecutive_failures += 1
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
