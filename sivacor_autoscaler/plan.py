"""What the fleet should look like, as a pure function.

All of the arithmetic and every guardrail lives here, deliberately free of Redis,
OpenStack and Girder, because this is the component whose bugs cost money: an
over-eager loop burns allocation, an under-eager one deadlocks submissions. Keeping
it pure means the interesting cases are cheap to test.

The caller (``controller``) gathers a :class:`FleetState`, asks :func:`decide` what to
do, and executes the answer.
"""

from __future__ import annotations

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
    spent: frozenset[str] = frozenset()
    #: Ids of instances whose celery worker started and reached the broker (D9).
    #: Only consulted when :attr:`Limits.provision_deadline` is set; see there for
    #: why that switch defaults to off.
    ready: frozenset[str] = frozenset()
    #: Consecutive instances that came up but never registered with celery.
    consecutive_failures: int = 0
    now: datetime | None = None


@dataclass(frozen=True)
class Decision:
    create: int = 0
    delete: tuple[str, ...] = ()
    #: Human-readable justification for every number above. Logged verbatim: this is
    #: the component whose decisions need to be auditable after the fact.
    reasons: tuple[str, ...] = field(default_factory=tuple)


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
    instance.

    **Deletes are never gated on the breaker.** A tripped breaker means "stop spending
    money", so continuing to reap is the whole point; skipping it would leak the very
    instances that tripped it.
    """
    reasons: list[str] = []
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
            # A spent instance is never written off, whatever the markers say: it
            # claimed a submission, which is positive proof its celery worked. The
            # readiness write is best-effort on the worker side, so trusting it over
            # an observed claim would be trusting the weaker signal.
            if inst.created_at is None or inst.id in state.ready or inst.id in state.spent:
                continue
            if now - inst.created_at > limits.provision_deadline:
                failed_to_provision.add(inst.id)

    # --- deletes -----------------------------------------------------------
    delete: list[str] = []
    seen: set[str] = set()

    def _reap(inst, why: str) -> None:
        # Guarded against double-listing: an instance can be both over the lifetime
        # ceiling and unprovisioned, and asking OpenStack to delete it twice turns a
        # tidy reap into a 404 on the second call.
        if inst.id in seen:
            return
        seen.add(inst.id)
        delete.append(inst.id)
        reasons.append(f"reap {inst.name}: {why}")

    for inst in state.instances:
        if inst.status == "SHUTOFF":
            _reap(inst, "SHUTOFF, work finished")
    for inst in live:
        if inst.id in failed_to_provision:
            _reap(
                inst,
                f"live for {now - inst.created_at} with no readiness marker, past "
                f"the {limits.provision_deadline} provisioning deadline; it can "
                "neither work nor reclaim itself",
            )
        if inst.created_at is None:
            continue
        age = now - inst.created_at
        if age > limits.max_lifetime:
            _reap(
                inst,
                f"live for {age}, over the {limits.max_lifetime} ceiling; "
                "supervisor did not fire",
            )

    # --- creates -----------------------------------------------------------
    # Instances remaining after this round's deletions still occupy their slots
    # until OpenStack actually removes them, so count conservatively against the cap.
    available = [
        i for i in live if i.id not in state.spent and i.id not in failed_to_provision
    ]
    shortfall = state.queue_depth - len(available)
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
            f"no new instances: depth={state.queue_depth}, {len(available)} "
            f"available of {len(live)} live ({spent_live} spent{dead}, "
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
            f"create {create}: depth={state.queue_depth}, only {len(available)} "
            f"available of {len(live)} live ({spent_live} spent{dead}, "
            f"serving={state.serving})"
        )

    return Decision(create=create, delete=tuple(delete), reasons=tuple(reasons))


def _tz_of(live):
    """Use an instance's own tzinfo so ages compare without a naive/aware TypeError.

    OpenStack hands back tz-aware timestamps; ``datetime.now()`` without a tz does
    not, and subtracting the two raises. The same trap the server-side reaper hit.
    """
    for inst in live:
        if inst.created_at is not None and inst.created_at.tzinfo is not None:
            return inst.created_at.tzinfo
    return None
