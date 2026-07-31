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


@dataclass(frozen=True)
class FleetState:
    #: Submissions published to the dispatch queue that no worker has taken yet.
    queue_depth: int
    #: Submissions currently executing. Load-bearing -- see :func:`decide`.
    serving: int
    instances: tuple[Instance, ...] = ()
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

    **Why ``serving`` is not optional.** The tempting formula is
    ``create = depth - live``: never over-provision, since live instances will pick up
    the queue. That deadlocks. An ephemeral worker drops its consumer on the dispatch
    queue the moment it accepts a submission (P3.2), so a *busy* instance will never
    take another one -- and it powers off when done. With two busy instances and one
    queued submission, ``depth - live`` is ``-1``, the controller creates nothing, and
    nothing will ever pick that submission up. Counting the busy ones separately is
    what avoids that:

        desired = depth + serving      # what the fleet must eventually cover
        create  = desired - live       # clamped to the cap

    **Treat this as accurate, not exact.** ``cancel_consumer`` is asynchronous and
    submissions are acked on receipt, so a worker can occasionally take a second one.
    That self-corrects -- the extra submission runs after the first, and the next loop
    sees the truth -- but it means the arithmetic is a good estimator rather than an
    invariant, and nothing downstream should assume one submission per instance.

    **Deletes are never gated on the breaker.** A tripped breaker means "stop spending
    money", so continuing to reap is the whole point; skipping it would leak the very
    instances that tripped it.
    """
    reasons: list[str] = []
    live = [i for i in state.instances if i.is_live]
    now = state.now or datetime.now(tz=_tz_of(live))

    # --- deletes -----------------------------------------------------------
    delete: list[str] = []
    for inst in state.instances:
        if inst.status == "SHUTOFF":
            delete.append(inst.id)
            reasons.append(f"reap {inst.name}: SHUTOFF, work finished")
    for inst in live:
        if inst.created_at is None:
            continue
        age = now - inst.created_at
        if age > limits.max_lifetime:
            delete.append(inst.id)
            reasons.append(
                f"reap {inst.name}: live for {age}, over the "
                f"{limits.max_lifetime} ceiling; supervisor did not fire"
            )

    # --- creates -----------------------------------------------------------
    # Instances remaining after this round's deletions still occupy their slots
    # until OpenStack actually removes them, so count conservatively against the cap.
    desired = state.queue_depth + state.serving
    shortfall = desired - len(live)
    headroom = limits.max_instances - len(live)
    create = max(0, min(shortfall, headroom))

    if state.consecutive_failures >= limits.breaker_threshold:
        if create:
            reasons.append(
                f"BREAKER OPEN: {state.consecutive_failures} consecutive instances "
                f"failed to register; refusing to create {create}. Deletes continue."
            )
        create = 0
    elif shortfall <= 0:
        reasons.append(
            f"no new instances: depth={state.queue_depth} + serving={state.serving} "
            f"= {desired}, already {len(live)} live"
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
            f"create {create}: depth={state.queue_depth} + serving={state.serving} "
            f"= {desired} needed, {len(live)} live"
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
