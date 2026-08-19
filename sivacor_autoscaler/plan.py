"""What the fleet should look like, as a pure function.

All of the arithmetic and every guardrail lives here, deliberately free of Redis,
OpenStack and Girder, because this is the component whose bugs cost money: an
over-eager loop burns allocation, an under-eager one deadlocks submissions. Keeping
it pure means the interesting cases are cheap to test.

The caller (``controller``) gathers a :class:`FleetState`, asks :func:`decide` what to
do, and executes the answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Instance:
    """One worker VM, as much of it as the decision needs."""

    id: str
    name: str
    status: str
    created_at: datetime | None = None

    @property
    def is_live(self) -> bool:
        """Booting or running, i.e. occupying an instance slot."""
        return self.status in ("BUILD", "ACTIVE")


@dataclass(frozen=True)
class RunningJob:
    """A submission Girder still believes is executing on a given instance.

    Carried purely so a reap can be *described* correctly. It changes no arithmetic:
    a SHUTOFF instance is deleted either way, because leaving it costs a quota slot
    and the submission is already unrecoverable -- its workspace died with the VM.
    What it changes is whether the operator can tell the two cases apart afterwards.
    """

    id: str
    #: ``meta.heartbeat``, the server's liveness signal. The gap between this and the
    #: poweroff is the whole diagnosis: ~17 min means the worker's idle supervisor
    #: powered the VM off after 8 unreachable ticks, i.e. celery died under a live run.
    heartbeat: datetime | None = None


@dataclass(frozen=True)
class WaitingSubmission:
    """A RUNNING submission that has not been given a worker yet.

    Absence of ``meta.worker_queue`` is the marker. Under targeted assignment (S2 of
    ``worker_sizing_plan.md``) the *controller* writes that field, so this set is both
    the assigner's input and, unchanged, the demand signal that already scales the
    fleet -- one query, one representation. See :attr:`FleetState.waiting`.

    Ages rather than timestamps because Girder stores ``created`` naive while ``now``
    here is tz-aware; :func:`signals.waiting_submissions` does the subtraction where
    that convention is known.
    """

    #: The Girder job id, as a string. Carried so an assignment can name it; the
    #: arithmetic uses it only as a stable tiebreaker.
    id: str
    age: timedelta

    #: Advertised RAM the submission asked for, per ``meta.requested_memory_gb``.
    #: ``None`` until P3 teaches the controller to read it -- at P2 the catalogue
    #: holds one rung, so every submission fits every instance and the field is
    #: deliberately unused by :func:`decide`.
    memory_gb: int | None = None

    #: Whether this submission is *ours* to place, per ``meta.awaiting_assignment``.
    #:
    #: **Demand and assignability part company here, and only here.** Girder records
    #: which route each submission took at submit time, so flipping the arm flag while
    #: one is in flight cannot make the controller assign something ``submit_job``
    #: already published to the shared queue -- two workers on one workspace. During
    #: rollout step 3 both kinds are live at once: a dispatched-but-unclaimed
    #: submission is still real demand and still needs an instance created for it, it
    #: simply must not be *placed* by us.
    #:
    #: Defaults to ``True`` so the arithmetic tests stay about arithmetic;
    #: :func:`signals.waiting_submissions` maps a missing field to ``False``, which is
    #: the safe direction for a job document written before this existed.
    assignable: bool = True


@dataclass(frozen=True)
class Limits:
    #: Hard ceiling on worker instances. Configure it BELOW the OpenStack quota: the
    #: same 25 instances carry the manager, the test mirror and any hand-made debug
    #: VM, so deriving this from the quota guarantees a collision with ordinary work.
    #: 5-10 is right for the pilot; AEA volume is nowhere near the quota, and the
    #: value of one-VM-per-submission is scale-to-zero, not peak throughput.
    max_instances: int = 5

    #: Delete a live instance older than this no matter what it claims to be doing.
    #: A safety net against a self-shutdown supervisor that never fires. Must stay
    #: comfortably above the server's own max-runtime cap, or the controller will
    #: delete instances the reaper still considers healthy -- and every boot now
    #: includes a cold image pull, so this covers pull + run, not run alone.
    max_lifetime: timedelta = timedelta(hours=30)

    #: Stop creating after this many consecutive instances fail to register. Without
    #: it a broken image loops forever, burning ~8 SU/hr per stuck instance.
    breaker_threshold: int = 3

    #: How long an instance may take to announce readiness before it is written off
    #: as failed to provision: excluded from available capacity, then deleted (D9).
    #:
    #: **``None`` disables the check entirely, and that is the default on purpose.**
    #: Enabling it against a worker image that does not write readiness markers
    #: deletes every healthy instance the moment it passes the deadline -- the same
    #: rollout-ordering hazard as P4.5's tro-utils floor. Turn it on only once the
    #: deployed image is known to announce, and confirm with
    #: ``redis-cli KEYS 'sivacor:ready:*'`` after one worker has come up.
    #:
    #: Sizing: boot->ready has measured 122-140 s across five runs, so 10 minutes is
    #: roughly 4x margin. The asymmetry favours generosity -- deleting a healthy but
    #: slow instance costs one boot (~2.5 min) and the submission is re-created for,
    #: while keeping a dead one costs a stalled submission plus up to 30 h of SUs.
    provision_deadline: timedelta | None = None

    #: How long a RUNNING submission may sit unclaimed before it counts as demand in
    #: its own right, independently of queue depth. See :attr:`FleetState.waiting`.
    #:
    #: **Ignored entirely under :attr:`assign`**, where the race it skips cannot
    #: happen: nothing is published until an instance has been chosen, so a waiting
    #: submission is never a submission already being started.
    #:
    #: **The grace exists to skip one transient, not to be cautious.** A worker that
    #: has just reserved the head task is briefly both absent from the queue and
    #: unclaimed -- from the moment celery hands it the message to the moment
    #: ``prepare_submission`` calls ``claim()``, a second or two. Counting that would
    #: create an instance for a submission already being started, on every submit.
    #: Anything comfortably past it is a genuine stall: nothing legitimately takes
    #: minutes between reserving a message and claiming it, because ``claim()`` is
    #: deliberately the first thing that task does.
    #:
    #: Two minutes is ~4 controller rounds at the 30 s tick, so a stall is corrected
    #: in well under the 30 min the server-side reaper would take to fail the
    #: submission outright.
    unclaimed_grace: timedelta = timedelta(minutes=2)

    #: Arm targeted assignment: choose an instance per submission and publish the
    #: chain to that instance's private queue (S2/S3 of ``worker_sizing_plan.md``).
    #:
    #: **Off by default, and it must be flipped together with Girder's own flag, never
    #: alone.** With this on while ``submit_job`` still publishes to the shared queue,
    #: a submission is both dispatched and assigned -- two workers, one workspace.
    #: With it off while ``submit_job`` has stopped publishing, nothing ever reaches a
    #: worker and the fleet looks like a healthy idle system while every submission
    #: waits. That is why the flag is one value read by both processes rather than two
    #: settings that can disagree; see P2's rollout order.
    #:
    #: Arming it also makes :attr:`FleetState.ready` load-bearing -- an unready
    #: instance is never assigned to -- so the caller must populate ``ready`` whenever
    #: this is on, independently of :attr:`provision_deadline`.
    assign: bool = False

    #: How old an instance may be and still receive its *first* assignment.
    #:
    #: The hazard: an instance nobody has assigned anything to is idle for its whole
    #: life, so it powers itself off on its own schedule. Hand it work seconds before
    #: that and the chain lands in a queue whose consumer disappears -- and because
    #: ``meta.worker_queue`` is now set, the submission has left the waiting set and
    #: become demand nothing can see, until the server-side reaper fails it up to 30
    #: minutes later.
    #:
    #: **Sized against ``BOOT_GRACE_SEC``, not the idle timeout.** The supervisor
    #: refuses to power off before 600 s of uptime (``worker-cloud-init.sh:86``), which
    #: floors the earliest poweroff at 10 minutes even though the 300 s idle clock has
    #: long expired -- checked on a 2 min timer, so in practice 600-720 s. Eight minutes
    #: therefore stops assigning ~2 min before the first moment a poweroff is possible,
    #: and leaves a ~5.5 min window (11 ticks) from the 122-140 s boot->ready.
    #:
    #: Refusing is the cheap direction -- the submission stays waiting, so it still
    #: counts as demand and a fresh instance is created for it -- which is why a flat
    #: ceiling is enough. Aged-out instances are excluded from *capacity* as well, or
    #: the fleet deadlocks on capacity it will never assign to.
    assign_max_age: timedelta = timedelta(minutes=8)


@dataclass(frozen=True)
class FleetState:
    #: Submissions published to the dispatch queue that no worker has taken yet.
    queue_depth: int
    #: Submissions currently executing. Reported for observability; the creation
    #: arithmetic uses :attr:`spent` instead -- see :func:`decide`.
    serving: int
    instances: tuple[Instance, ...] = ()
    #: Ids of instances that have already claimed a submission and therefore no
    #: longer consume the dispatch queue. Empty is safe: the arithmetic degrades to
    #: the older, stall-prone ``depth + serving`` estimate rather than misbehaving.
    #:
    #: Under :attr:`Limits.assign` the *controller* writes that marker, so this means
    #: "has been given a submission" rather than "has started one". The D9 skip below
    #: still rests on evidence that celery worked, but by a different route: only a
    #: :attr:`ready` instance is ever assigned to.
    spent: frozenset[str] = frozenset()
    #: Ids of instances whose celery worker started and reached the broker (D9).
    #: Consulted when :attr:`Limits.provision_deadline` is set -- see there for why
    #: that switch defaults to off -- **and** whenever :attr:`Limits.assign` is on,
    #: because an instance that has not registered is never assigned work.
    ready: frozenset[str] = frozenset()
    #: Every RUNNING submission that no worker has been given -- demand that
    #: :attr:`queue_depth` cannot see, because a reserved-but-unstarted message has
    #: already left the Redis list. Empty degrades to the depth-only reading, i.e. the
    #: behaviour before it existed. See :func:`signals.waiting_submissions`.
    #:
    #: This is *also* the assigner's work list under :attr:`Limits.assign`, and
    #: deliberately the same field: the demand signal and the assignment input are the
    #: same Mongo query, so representing them twice would let the fleet size itself
    #: from one reading and place work from another.
    waiting: tuple[WaitingSubmission, ...] = ()
    #: Consecutive instances that came up but never registered with celery.
    consecutive_failures: int = 0
    #: Instance id -> the submission Girder still has RUNNING on it. Diagnostics only;
    #: an empty mapping degrades every reap message to its old, less specific wording
    #: and nothing else. See :func:`signals.running_jobs_by_instance`.
    running_jobs: Mapping[str, RunningJob] = field(default_factory=dict)
    now: datetime | None = None


@dataclass(frozen=True)
class Decision:
    create: int = 0
    delete: tuple[str, ...] = ()
    #: ``(submission id, instance id)`` pairs to bind, oldest submission first (S7).
    #: The caller claims each submission atomically and *then* publishes its chain to
    #: that instance's private queue -- never the reverse order, which can publish the
    #: same chain twice if the tick dies between the two. Empty unless
    #: :attr:`Limits.assign` is on.
    assign: tuple[tuple[str, str], ...] = ()
    #: Human-readable justification for every number above. Logged verbatim: this is
    #: the component whose decisions need to be auditable after the fact.
    reasons: tuple[str, ...] = field(default_factory=tuple)
    #: A **subset of** :attr:`reasons`: the ones describing a fleet that is not
    #: behaving as designed, and so deserving WARNING rather than INFO. A severity view
    #: rather than a separate bucket, so :attr:`reasons` stays the complete audit trail
    #: it is documented to be, and the normal chatter -- one "no new instances" line
    #: every 30 s -- still cannot bury an anomaly.
    alerts: tuple[str, ...] = field(default_factory=tuple)
    #: Ids whose deletion destroys evidence worth keeping: capture before deleting.
    #: A worker VM is the only place its own journal and console buffer exist, and
    #: ``delete_server`` is irreversible, so this is the last moment either can be read.
    abnormal: frozenset[str] = frozenset()
    #: Instance id -> why it is being reaped, the same text as in :attr:`reasons` or
    #: :attr:`alerts` without the ``reap <name>:`` prefix. Carried by id so the caller
    #: does not have to recover the association by parsing log lines back apart.
    reap_reasons: Mapping[str, str] = field(default_factory=dict)


def decide(state: FleetState, limits: Limits) -> Decision:
    """Return the create/delete actions the fleet needs.

    **Count available instances, not all live ones.** The tempting formula is
    ``create = depth - live``: never over-provision, since live instances will pick up
    the queue. That deadlocks. An ephemeral worker drops its consumer on the dispatch
    queue the moment it accepts a submission (P3.2), so a *spent* instance will never
    take another one -- and it powers off when done. With two spent instances and one
    queued submission, ``depth - live`` is ``-1``, the controller creates nothing, and
    nothing ever picks that submission up.

    The original fix estimated around it with ``desired = depth + serving``, which
    deadlocked less but still stalled: ``serving`` counts submissions *executing*, and
    an instance that has finished its submission but not yet powered off is neither
    serving nor available, yet still counts as ``live``. Measured 2026-08-01: a
    submission waited **4 min 45 s** behind exactly that, and would have waited ~13 min
    had a later submission not bumped the depth (D8, P3.5).

    So ask the question directly instead of estimating it::

        available = live instances that have NOT claimed a submission
        create    = depth - available          # clamped to the cap

    ``spent`` comes from a marker the worker writes to Girder when it claims a
    submission, not from a broker broadcast -- see :func:`signals.spent_instance_ids`
    for why that distinction matters.

    **Treat this as accurate, not exact.** A worker that becomes ready while two or
    more submissions are queued takes *two*: celery delivers them in one prefetch,
    milliseconds after startup and before ``cancel_consumer`` can run. The spare
    instance simply idles and is reaped, so this over-provisions slightly rather than
    stalling -- the safe direction. Nothing downstream may assume one submission per
    instance. (Under :attr:`Limits.assign` there is no shared queue to prefetch from,
    which is what makes one-per-instance true rather than hoped for.)

    **Deletes are never gated on the breaker.** A tripped breaker means "stop spending
    money", so continuing to reap is the whole point; skipping it would leak the very
    instances that tripped it.

    **Assignment is the same decision as capacity, which is why it lives here**
    (S3 of ``worker_sizing_plan.md``). Under :attr:`Limits.assign` the function also
    returns which waiting submission goes on which instance, oldest submission first
    (S7). Putting it anywhere else would mean implementing that ordering twice, from two
    snapshots taken at two different times, and the failure of the two disagreeing is
    over-provisioning -- silent, and paid for in SUs. Here it is answerable by a unit
    test with no OpenStack, no Mongo and no clock.

    Capacity and assignability are deliberately *different* sets. A booting instance is
    capacity, because it will serve; only an instance that has registered with the
    broker is assignable, because a chain published to a queue that never gets a
    consumer is a stall no signal can see. Instances past
    :attr:`Limits.assign_max_age` are excluded from both -- one that will never be
    assigned to must not absorb demand.
    """
    reasons: list[str] = []
    alerts: list[str] = []
    live = [i for i in state.instances if i.is_live]
    now = state.now or datetime.now(tz=_tz_of(live))

    # --- instances that never provisioned (D9) -----------------------------
    # A VM that boots but fails to provision never claims and never serves, so
    # without this it reads as available capacity forever -- and it cannot power
    # itself off either, because the supervisor is written by the script that
    # failed. Absence of a readiness marker past a boot deadline is what separates
    # that from "still booting" and from "healthy and idle".
    failed_to_provision: set[str] = set()
    if limits.provision_deadline is not None:
        for inst in live:
            # A spent instance is never written off, whatever the markers say: it holds
            # a submission, which is positive proof its celery worked -- by claiming it,
            # or by having been ready when the assigner chose it. The readiness write is
            # best-effort, so trusting it over that would be trusting the weaker signal.
            if inst.created_at is None or inst.id in state.ready or inst.id in state.spent:
                continue
            if now - inst.created_at > limits.provision_deadline:
                failed_to_provision.add(inst.id)

    # --- deletes -----------------------------------------------------------
    delete: list[str] = []
    seen: set[str] = set()
    abnormal: set[str] = set()
    reap_reasons: dict[str, str] = {}

    def _reap(inst, why: str, *, alarming: bool = False) -> None:
        # Guarded against double-listing: an instance can be both over the lifetime
        # ceiling and unprovisioned, and asking OpenStack to delete it twice turns a
        # tidy reap into a 404 on the second call.
        if inst.id in seen:
            return
        seen.add(inst.id)
        delete.append(inst.id)
        reap_reasons[inst.id] = why
        line = f"reap {inst.name}: {why}"
        reasons.append(line)
        if alarming:
            alerts.append(line)
            abnormal.add(inst.id)

    for inst in state.instances:
        if inst.status != "SHUTOFF":
            continue
        # "SHUTOFF, work finished" was this branch's only wording until 2026-08-11,
        # and it was an unchecked assumption. Three production submissions were reaped
        # for no heartbeat on 2026-08-10/11; in all three the worker had powered itself
        # off *mid-run* (celery unreachable for 8 ticks) and this loop deleted it ~17
        # min after the last heartbeat, logging "work finished" over the top. Since a
        # delete also destroys the console buffer and the journal, that one word was
        # the difference between a diagnosable failure and an unexplainable one.
        if job := state.running_jobs.get(inst.id):
            _reap(
                inst,
                f"SHUTOFF while submission {job.id} is still RUNNING"
                f"{_since(job.heartbeat, now)}. The worker did NOT finish its work: "
                "it powered off under a live run, so its own logs are the only record "
                "of why -- capture them before this delete",
                alarming=True,
            )
        else:
            _reap(inst, "SHUTOFF, work finished")
    for inst in live:
        if inst.id in failed_to_provision:
            _reap(
                inst,
                f"live for {now - inst.created_at} with no readiness marker, past "
                f"the {limits.provision_deadline} provisioning deadline; it can "
                "neither work nor reclaim itself",
                alarming=True,
            )
        if inst.created_at is None:
            continue
        age = now - inst.created_at
        if age > limits.max_lifetime:
            _reap(
                inst,
                f"live for {age}, over the {limits.max_lifetime} ceiling; "
                "supervisor did not fire",
                alarming=True,
            )

    # --- assignment (S2) ---------------------------------------------------
    # After the deletes, so a reaped instance is never handed work; before the creates,
    # because an instance nothing will be assigned to is not capacity either.
    assign: list[tuple[str, str]] = []
    aged_out: set[str] = set()
    if limits.assign:
        unassigned = [
            i
            for i in live
            if i.id not in state.spent
            and i.id not in failed_to_provision
            and i.id not in seen
        ]
        aged_out = {
            i.id
            for i in unassigned
            if i.created_at is not None
            and now - i.created_at > limits.assign_max_age
        }
        # Assignment is pessimistic where capacity is optimistic: a booting instance
        # will serve, but only a broker-registered one is given work, because a chain
        # published to an instance that never provisions sets `meta.worker_queue` -- so
        # the submission leaves the waiting set and becomes demand no signal can see.
        # That requirement is also what ties arming this to a worker image that
        # announces readiness.
        #
        # Fleet instances only: the manager's static worker announces readiness too but
        # is not an OpenStack server, so it never appears in `live`. A known gap for
        # D6's oversized-package tail, which wants `sivacor.static-01` as a target.
        assignable = [
            i for i in unassigned if i.id not in aged_out and i.id in state.ready
        ]
        # Youngest first: an unassigned instance has been idle its whole life, so the
        # youngest is furthest from its own poweroff. See assign_max_age.
        assignable.sort(key=_youngest_first(now))
        # Ours to place, per WaitingSubmission.assignable: during rollout step 3 a
        # submission Girder already dispatched is still demand, but publishing a second
        # chain for it would put two workers on one workspace.
        placeable = tuple(w for w in _oldest_first(state.waiting) if w.assignable)
        # Oldest submission first (S7). One size means everything fits everywhere, so a
        # plain zip; P3 adds the size filter and the head-of-line stop.
        for sub, inst in zip(placeable, assignable):
            assign.append((sub.id, inst.id))
            reasons.append(
                f"assign submission {sub.id} -> {inst.name}: waiting {sub.age}, "
                f"oldest of {len(placeable)} placeable ({len(state.waiting)} waiting)"
            )
        if aged_out and state.waiting:
            # Only worth saying in this combination. Ageing out with nothing waiting is
            # routine -- an over-provisioned instance idling towards its own poweroff --
            # but work waiting while an instance sits unassignable is never a busy
            # fleet.
            line = (
                f"{len(aged_out)} instance(s) too old to assign while "
                f"{len(state.waiting)} submission(s) wait: past "
                f"{limits.assign_max_age} they may power themselves off at any tick, "
                "so they are excluded from capacity too and replaced ("
                + ", ".join(sorted(i.name for i in live if i.id in aged_out))
                + ")"
            )
            reasons.append(line)
            alerts.append(line)
        elif placeable and not assignable:
            # Routine on an empty fleet -- the tick before the instances exist -- but
            # it is the only line that distinguishes that from the arming failure where
            # nothing is ever assignable because no worker announces readiness.
            reasons.append(
                f"{len(placeable)} submission(s) waiting, none assignable: "
                f"{len(live)} live, {len([i for i in live if i.id in state.spent])} "
                f"already assigned, {len([i for i in live if i.id in state.ready])} "
                f"registered with the broker"
            )
        elif state.waiting and not placeable:
            # The mixed-mode reading, and it must not look like the line above: these
            # submissions are not stuck, they are on the other path and a worker is
            # already coming for them. Only reachable during rollout step 3.
            reasons.append(
                f"{len(state.waiting)} submission(s) waiting, none of them ours to "
                "place: dispatched to the shared queue before assignment was armed"
            )

    # --- creates -----------------------------------------------------------
    # Instances remaining after this round's deletions still occupy their slots
    # until OpenStack actually removes them, so count conservatively against the cap.
    # `aged_out` (empty unless assignment is armed) must go here as well as out of
    # `assignable`: capacity nothing will ever be assigned to absorbs demand forever,
    # which is the run-3/run-5 stall through a new door.
    available = [
        i
        for i in live
        if i.id not in state.spent
        and i.id not in failed_to_provision
        and i.id not in aged_out
    ]
    # Demand is the *larger* of the two readings, never their sum: a submission
    # waiting in the queue is also RUNNING-and-unclaimed in Girder, so adding them
    # would double-count every ordinary submit and provision twice over. Taking the
    # max means depth still drives the normal case, while a submission depth has lost
    # sight of -- reserved by a worker that cannot start it -- still gets an instance.
    if limits.assign:
        # No grace: it exists to skip the reserve-to-claim window, which S2 removes
        # outright, and keeping it would delay every submission by two minutes. Still
        # max() against depth, because rollout step 3 runs both paths and a dispatched
        # submission is also unassigned.
        stalled = state.waiting
        demand = max(state.queue_depth, len(state.waiting))
    else:
        stalled = tuple(w for w in state.waiting if w.age > limits.unclaimed_grace)
        demand = max(state.queue_depth, len(stalled))
    if not limits.assign and len(stalled) > state.queue_depth:
        # The dispatch queue is not showing work that Girder says is unserved. An
        # anomaly *while submissions are dispatched blind*: either a worker is holding a
        # message it will not run, or a submission was published to a queue nobody
        # consumes. Hence the flag in the condition -- once assignment is armed the same
        # reading is the design, and would fire on every tick of a healthy fleet.
        # Appended to BOTH lists because `alerts` is documented as a severity view
        # *over* `reasons`, not a second bucket -- an alert missing from the audit trail
        # would be a bug.
        line = (
            f"{len(stalled)} submission(s) unclaimed for over {limits.unclaimed_grace} "
            f"but depth={state.queue_depth}: the dispatch queue has lost sight of "
            "work Girder still considers unserved; scaling on the unclaimed count"
        )
        reasons.append(line)
        alerts.append(line)
    shortfall = demand - len(available)
    headroom = limits.max_instances - len(live)
    create = max(0, min(shortfall, headroom))
    spent_live = len([i for i in live if i.id in state.spent])
    # Surfaced only when non-zero: on a healthy fleet it is noise on every tick, and
    # when it is non-zero it is the first thing worth seeing.
    dead = (
        f", {len(failed_to_provision)} unprovisioned" if failed_to_provision else ""
    )

    if state.consecutive_failures >= limits.breaker_threshold:
        if create:
            reasons.append(
                f"BREAKER OPEN: {state.consecutive_failures} consecutive instances "
                f"failed to register; refusing to create {create}. Deletes continue."
            )
        create = 0
    elif shortfall <= 0:
        reasons.append(
            f"no new instances: depth={state.queue_depth}{_unclaimed_note(stalled)}, "
            f"{len(available)} available of {len(live)} live ({spent_live} spent{dead}, "
            f"serving={state.serving})"
        )
    elif create < shortfall:
        # Not an error: the queue simply waits. Logged loudly because a cap that
        # silently throttles looks identical to a controller that has stopped working.
        reasons.append(
            f"CAPPED: want {shortfall} more but only {headroom} slot(s) left of "
            f"max_instances={limits.max_instances}; {shortfall - create} submission(s) "
            "will wait"
        )
    else:
        reasons.append(
            f"create {create}: depth={state.queue_depth}{_unclaimed_note(stalled)}, "
            f"only {len(available)} available of {len(live)} live "
            f"({spent_live} spent{dead}, serving={state.serving})"
        )

    return Decision(
        create=create,
        delete=tuple(delete),
        assign=tuple(assign),
        reasons=tuple(reasons),
        alerts=tuple(alerts),
        abnormal=frozenset(abnormal),
        reap_reasons=reap_reasons,
    )



def _oldest_first(waiting) -> tuple[WaitingSubmission, ...]:
    """Waiting submissions in the order S7 says they must be served.

    Oldest first, and the id as a tiebreaker so two submissions created in the same
    millisecond still order deterministically -- a decision that changes between ticks
    for no observable reason is one nobody can debug from the log.
    """
    return tuple(sorted(waiting, key=lambda w: (-w.age, w.id)))


def _youngest_first(now):
    """Sort key putting the instance with the most life left in it first.

    A worker's supervisor refuses to power off before ``BOOT_GRACE_SEC`` of *uptime*, so
    for an unassigned instance the time left is a direct function of its age: youngest
    is furthest from poweroff. See ``Limits.assign_max_age``.

    An undated instance sorts last rather than first. It was probably created moments
    ago, but it is also the one whose age cannot be checked against the ceiling, and
    preferring a candidate because less is known about it is the wrong instinct.
    """

    def key(inst):
        if inst.created_at is None:
            return (1, timedelta(0), inst.id)
        # Smallest age first, i.e. youngest first. The leading 0/1 is what keeps the
        # undated instance behind every dated one regardless of its age.
        return (0, now - inst.created_at, inst.id)

    return key


def _unclaimed_note(stalled) -> str:
    """``, unclaimed=2`` when submissions are stalled, nothing when none are.

    Omitted on a healthy fleet for the same reason as the ``unprovisioned`` note: a
    field that is always zero trains the reader to skip the line it appears on.
    """
    return f", unclaimed={len(stalled)}" if stalled else ""


def _since(heartbeat, now) -> str:
    """`` (last heartbeat 0:17:12 ago)``, or nothing if there is no heartbeat.

    Naive/aware mismatches are possible here in a way they are not for instance
    timestamps: ``meta.heartbeat`` comes back from pymongo naive by default, while
    ``now`` follows OpenStack and is aware. Subtracting those raises -- the same trap
    ``_tz_of`` exists for, and one that must not be allowed to break a reap.
    """
    if heartbeat is None:
        return ""
    try:
        return f" (last heartbeat {now - heartbeat} ago)"
    except TypeError:
        return f" (last heartbeat {heartbeat.isoformat()})"


def _tz_of(live):
    """Use an instance's own tzinfo so ages compare without a naive/aware TypeError.

    OpenStack hands back tz-aware timestamps; ``datetime.now()`` without a tz does
    not, and subtracting the two raises. The same trap the server-side reaper hit.
    """
    for inst in live:
        if inst.created_at is not None and inst.created_at.tzinfo is not None:
            return inst.created_at.tzinfo
    return None
