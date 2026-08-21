"""Tests for the scaling arithmetic and its guardrails.

This is the component whose bugs cost money in one direction and deadlock
submissions in the other, so the cases below are the ones worth being sure about
rather than a sweep for coverage.
"""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from sivacor_autoscaler.plan import (
    Decision,
    FleetState,
    Instance,
    Limits,
    SizeSpec,
    WaitingSubmission,
    decide,
)

NOW = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
#: Shorthand for the P3 cases below, where what matters about an age is only that it
#: is comfortably past `unclaimed_grace` and orders predictably against its siblings.
MIN = timedelta(minutes=1)
LIMITS = Limits(max_instances=5, max_lifetime=timedelta(hours=30), breaker_threshold=3)


def inst(name, status="ACTIVE", age=timedelta(minutes=5), size=None):
    return Instance(
        id=f"id-{name}", name=name, status=status, created_at=NOW - age, size=size
    )


def state(
    depth=0,
    serving=0,
    instances=(),
    failures=0,
    spent=(),
    ready=(),
    unclaimed_ages=(),
    waiting=(),
):
    return FleetState(
        queue_depth=depth,
        serving=serving,
        instances=tuple(instances),
        spent=frozenset(f"id-{n}" for n in spent),
        ready=frozenset(f"id-{n}" for n in ready),
        # Two ways in for one field. `unclaimed_ages=` is how every test written before
        # assignment existed describes demand -- ages only, no identity, because the
        # scaling arithmetic never needed one -- and it is kept verbatim so those cases
        # keep documenting the incidents they were written for. `waiting=` is the same
        # field with the ids the assigner needs. Not both at once.
        waiting=tuple(waiting)
        or tuple(
            WaitingSubmission(id=f"sub-{i}", age=age)
            for i, age in enumerate(unclaimed_ages)
        ),
        consecutive_failures=failures,
        now=NOW,
    )


def waiting(*ages, ids=None, assignable=True, sizes=None):
    """Waiting submissions, named ``sub-0``, ``sub-1``, ... unless ``ids`` says otherwise.

    ``assignable=False`` is the mixed-mode submission: dispatched to the shared queue
    by ``submit_job`` before assignment was armed, so it is demand but not ours.

    ``sizes`` gives each submission a ``memory_gb``, positionally. Omitted leaves them
    all ``None``, which is every test written before P3 -- and, per
    ``WaitingSubmission.memory_gb``, resolves to the smallest catalogue rung.
    """
    names = ids or [f"sub-{i}" for i in range(len(ages))]
    mems = list(sizes or ()) + [None] * (len(ages) - len(sizes or ()))
    return tuple(
        WaitingSubmission(id=n, age=a, assignable=assignable, memory_gb=m)
        for n, a, m in zip(names, ages, mems)
    )


#: Limits with the D9 check armed. The production default is None (off), so every
#: test that wants the behaviour has to ask for it explicitly -- which is the point.
D9 = Limits(provision_deadline=timedelta(minutes=10))


def rungs(decision):
    """The memory rungs of a decision's creates, which is what most of these assert.

    ``Decision.create`` became a tuple of :class:`Create` pairs in C3, since a rung
    alone no longer says what to boot once volumes are per-submission. These assertions
    were always about rungs, so they stay about rungs; the disk half has its own tests
    in ``test_volume_headroom``.
    """
    return tuple(c.rung for c in decision.create)


def test_scale_from_zero():
    d = decide(state(depth=3), LIMITS)
    assert len(d.create) == 3


def test_nothing_queued_nothing_created():
    d = decide(state(depth=0, instances=[inst("w1")]), LIMITS)
    assert d == Decision(create=(), delete=(), reasons=d.reasons)
    assert any("no new instances" in r for r in d.reasons)


def test_spent_instances_do_not_absorb_the_queue():
    """
    The deadlock the spent/available distinction exists to prevent.

    Two instances have claimed submissions. Because an ephemeral worker drops its
    dispatch-queue consumer the moment it accepts one (P3.2), neither will ever take
    the queued third -- and both power off when done. A naive `depth - live` gives
    -1, creates nothing, and that submission waits forever.
    """
    d = decide(
        state(
            depth=1, serving=2, instances=[inst("w1"), inst("w2")], spent=["w1", "w2"]
        ),
        LIMITS,
    )

    assert len(d.create) == 1, "a queued submission with only spent workers must scale up"


def test_finished_but_not_yet_powered_off_does_not_block_creation():
    """
    The stall this replaced ``depth + serving`` to fix (D8 option C, P3.5).

    Measured 2026-08-01: a worker finished its submission and sat waiting out the
    boot grace plus idle clock. It was neither serving nor available, but still
    ``live`` -- so ``depth(1) + serving(0) = 1`` against ``1 live`` created nothing
    and the next submission waited 4 min 45 s. Alone it would have waited ~13 min.
    """
    spent_and_idle = inst("w1")
    d = decide(
        state(depth=1, serving=0, instances=[spent_and_idle], spent=["w1"]), LIMITS
    )

    assert len(d.create) == 1, "a spent-but-alive worker must not count as capacity"
    assert any("1 spent" in r for r in d.reasons)


def test_available_instances_are_not_double_counted():
    """The other direction: an idle worker that has claimed nothing will take the work."""
    d = decide(state(depth=1, instances=[inst("w1")]), LIMITS)

    assert len(d.create) == 0
    assert any("1 available" in r for r in d.reasons)


def test_booting_instances_are_not_double_counted():
    """An instance still in BUILD will take a queued submission; don't create twice."""
    d = decide(
        state(depth=2, serving=0, instances=[inst("w1", status="BUILD")]), LIMITS
    )
    assert len(d.create) == 1


def test_cap_is_respected_and_says_so():
    """A cap that throttles silently is indistinguishable from a broken controller."""
    live = [inst(f"w{i}") for i in range(5)]
    spent = [f"w{i}" for i in range(5)]
    d = decide(state(depth=4, serving=5, instances=live, spent=spent), LIMITS)

    assert len(d.create) == 0
    assert any("CAPPED" in r for r in d.reasons)
    assert any("will wait" in r for r in d.reasons)


def test_partial_cap_creates_what_it_can():
    live = [inst(f"w{i}") for i in range(4)]
    spent = [f"w{i}" for i in range(4)]
    d = decide(state(depth=3, serving=4, instances=live, spent=spent), LIMITS)
    assert len(d.create) == 1  # one slot left of five


def test_shutoff_instances_are_reaped():
    d = decide(state(instances=[inst("done", status="SHUTOFF")]), LIMITS)
    assert d.delete == ("id-done",)
    assert any("SHUTOFF" in r for r in d.reasons)


def test_shutoff_instances_do_not_count_against_the_cap():
    """They hold no work; counting them would throttle the fleet as they accumulate."""
    dead = [inst(f"d{i}", status="SHUTOFF") for i in range(5)]
    d = decide(state(depth=2, instances=dead), LIMITS)
    assert len(d.create) == 2
    assert len(d.delete) == 5


def test_instance_over_max_lifetime_is_deleted():
    old = inst("zombie", age=timedelta(hours=31))
    d = decide(state(instances=[old]), LIMITS)
    assert d.delete == ("id-zombie",)
    assert any("supervisor did not fire" in r for r in d.reasons)


def test_instance_under_max_lifetime_survives():
    d = decide(state(instances=[inst("busy", age=timedelta(hours=29))]), LIMITS)
    assert d.delete == ()


def test_breaker_stops_creation():
    d = decide(state(depth=5, failures=3), LIMITS)
    assert len(d.create) == 0
    assert any("BREAKER OPEN" in r for r in d.reasons)


def test_breaker_does_not_stop_reaping():
    """
    A tripped breaker means "stop spending money", so reaping must continue.

    Gating deletes on it would leak exactly the instances that tripped it -- the
    failure mode is a full quota and a fleet that can no longer boot anything.
    """
    d = decide(
        state(depth=5, failures=9, instances=[inst("dead", status="SHUTOFF")]), LIMITS
    )
    assert len(d.create) == 0
    assert d.delete == ("id-dead",)


def test_breaker_just_below_threshold_still_creates():
    d = decide(state(depth=1, failures=2), LIMITS)
    assert len(d.create) == 1


def test_naive_timestamps_do_not_raise():
    """
    OpenStack returns tz-aware times; datetime.now() does not, and mixing them raises.

    The server-side reaper hit exactly this, so it is worth an explicit test rather
    than a comment.

    ``created_at`` is relative to real time on purpose: ``now=None`` is the whole point
    of the case, so a hardcoded date silently becomes older than ``max_lifetime`` as the
    wall clock advances and the instance starts getting reaped for age instead. This
    test began failing on 2026-08-02, 30 h after the date it used to pin.
    """
    naive = Instance(
        id="n",
        name="n",
        status="ACTIVE",
        created_at=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    d = decide(
        FleetState(queue_depth=0, serving=0, instances=(naive,), now=None), LIMITS
    )
    assert d.delete == ()


def test_missing_created_at_is_not_reaped():
    """An instance OpenStack has not dated yet must not be treated as ancient."""
    undated = Instance(id="u", name="u", status="ACTIVE", created_at=None)
    d = decide(state(instances=[undated]), LIMITS)
    assert d.delete == ()


def test_every_decision_is_explained():
    """Decisions are logged verbatim; a number with no reason is unauditable."""
    d = decide(state(depth=2, serving=1, instances=[inst("w1")]), LIMITS)
    assert d.reasons and all(r.strip() for r in d.reasons)


# -- D9: instances that never provisioned ----------------------------------------
#
# Run 6 (2026-08-02): a VM lost the dpkg-lock race, aborted cloud-init under `set -e`
# before its systemd units were written, and so came up with neither celery nor a
# self-shutdown supervisor. It was ACTIVE, could never work, could never reclaim
# itself, and -- being neither spent nor serving -- was counted as available capacity
# for the whole run, stalling a submission behind it.


def test_unprovisioned_instance_is_reaped_and_not_counted_as_capacity():
    """The run-6 phantom: past the deadline, no readiness marker, nothing claimed."""
    dead = inst("dead", age=timedelta(minutes=30))
    d = decide(state(depth=1, instances=[dead]), D9)

    assert d.delete == ("id-dead",)
    # ...and a replacement is created in the same round, because the corpse no longer
    # absorbs the shortfall. Both halves matter: reaping alone would still stall.
    assert len(d.create) == 1
    assert any("no readiness marker" in r for r in d.reasons)


def test_ready_instance_is_normal_capacity():
    ready = inst("ready", age=timedelta(minutes=30))
    d = decide(state(depth=1, instances=[ready], ready=["ready"]), D9)

    assert d.delete == ()
    assert len(d.create) == 0


def test_instance_inside_the_deadline_is_left_alone():
    """Still booting is not the same as failed; boot->ready measured 122-140 s."""
    booting = inst("booting", age=timedelta(minutes=3))
    d = decide(state(depth=1, instances=[booting]), D9)

    assert d.delete == ()
    assert len(d.create) == 0


def test_spent_instance_is_never_written_off_even_without_a_marker():
    """A claimed submission is positive proof celery worked.

    The readiness write is best-effort on the worker side, so an observed claim is
    the stronger signal and must win. Reaping here would kill a running submission.
    """
    working = inst("working", age=timedelta(minutes=30))
    d = decide(state(depth=0, instances=[working], spent=["working"]), D9)

    assert d.delete == ()


def test_check_is_off_by_default():
    """Arming it against an image that does not announce would delete the fleet.

    The default must therefore be inert, not merely conservative -- this is the
    rollout-ordering hazard, and the only thing standing between a stale worker
    image and a controller that reaps every healthy instance it has.
    """
    unmarked = inst("unmarked", age=timedelta(hours=2))
    d = decide(state(depth=0, instances=[unmarked]), Limits())

    assert d.delete == ()


def test_unprovisioned_and_over_lifetime_is_deleted_once():
    """Both rules match; asking OpenStack to delete it twice 404s on the second."""
    ancient = inst("ancient", age=timedelta(hours=40))
    d = decide(state(instances=[ancient]), D9)

    assert d.delete == ("id-ancient",)


def test_unprovisioned_count_is_reported():
    dead = inst("dead", age=timedelta(minutes=30))
    d = decide(state(depth=1, instances=[dead]), D9)

    assert any("1 unprovisioned" in r for r in d.reasons)


# --- unclaimed submissions: demand queue depth cannot see ------------------
# Production 2026-08-12: a worker restarted mid-chain by the wedge supervisor
# re-subscribed to the dispatch queue, reserved a submission it could not start, and
# the controller read depth=0 with its only instance spent -- so it created nothing
# and the submission sat untouched. Every input was correct except depth.

STALE = timedelta(minutes=5)
FRESH = timedelta(seconds=3)


def test_unclaimed_submission_scales_up_when_depth_lost_it():
    spent_and_busy = inst("w1")
    d = decide(
        state(depth=0, serving=1, instances=[spent_and_busy], spent=["w1"],
              unclaimed_ages=[STALE]),
        LIMITS,
    )
    assert len(d.create) == 1, (
        "a submission unclaimed for minutes must scale up even at depth=0: its "
        "message is reserved by a worker that cannot start it"
    )


def test_unclaimed_alert_is_also_a_reason():
    """`alerts` is documented as a severity view over `reasons`, not a second bucket."""
    d = decide(state(depth=0, unclaimed_ages=[STALE]), LIMITS)
    assert d.alerts, "a queue that lost sight of unserved work is an anomaly"
    for a in d.alerts:
        assert a in d.reasons, "every alert must remain in the audit trail"


def test_unclaimed_within_grace_is_ignored():
    """The gap between reserving a message and calling claim() is normal."""
    d = decide(state(depth=0, unclaimed_ages=[FRESH]), LIMITS)
    assert len(d.create) == 0
    assert not d.alerts


def test_unclaimed_is_not_added_to_depth():
    """A queued submission is *also* unclaimed; counting both provisions twice."""
    d = decide(state(depth=1, unclaimed_ages=[STALE]), LIMITS)
    assert len(d.create) == 1, "max(depth, unclaimed), never depth + unclaimed"
    assert not d.alerts, "depth already accounts for it, so nothing is anomalous"


def test_unclaimed_does_not_double_provision_against_available_capacity():
    d = decide(state(depth=0, instances=[inst("w1")], unclaimed_ages=[STALE]), LIMITS)
    assert len(d.create) == 0, "an idle available worker will take it; no new instance"


def test_unclaimed_respects_the_breaker():
    d = decide(
        state(depth=0, unclaimed_ages=[STALE, STALE], failures=3),
        LIMITS,
    )
    assert len(d.create) == 0, "a tripped breaker must not be bypassed by a new signal"


def test_unclaimed_absent_keeps_old_behaviour():
    d = decide(state(depth=2), LIMITS)
    assert len(d.create) == 2
    assert not d.alerts


# --- targeted assignment (S2/S7 of worker_sizing_plan.md) ------------------
#
# The controller stops letting a shared queue decide which worker takes which
# submission and binds them itself, one instance per submission. The arithmetic is
# driven here, in full, before any of it touches Mongo -- because the failure mode of
# getting it wrong is a submission that waits forever while the fleet reports itself
# healthy and idle, which is the same shape as every stall in autoscaling_plan.md.

#: Assignment armed. Everything else default, so these cases exercise the flag alone.
ASSIGN = Limits(max_instances=5, assign=True)

OLDEST = timedelta(minutes=9)
OLDER = timedelta(minutes=6)
RECENT = timedelta(minutes=1)


def test_assignment_is_off_by_default():
    """The flag must be inert until Girder stops dispatching, or work runs twice.

    With this on while ``submit_job`` still publishes to the shared queue, a submission
    is both dispatched and assigned: two workers, one workspace. The default is
    therefore not merely conservative, it is the only safe half of the pair.
    """
    d = decide(state(waiting=waiting(OLDEST), instances=[inst("w1")], ready=["w1"]), LIMITS)

    assert d.assign == ()


def test_the_oldest_waiting_submission_is_assigned_first():
    """S7, and the whole reason ordering is a decision rather than an accident.

    Skipping the head of the line raises utilisation and starves whichever submission
    is hardest to place -- at one size that is nobody, but the rule has to be right
    before P3 makes it load-bearing, and a researcher watching later submissions run
    ahead of theirs cannot be given an explanation for it.
    """
    d = decide(
        state(
            waiting=waiting(RECENT, OLDEST, ids=["young", "old"]),
            instances=[inst("w1")],
            ready=["w1"],
        ),
        ASSIGN,
    )

    assert d.assign == (("old", "id-w1"),)


def test_scarce_capacity_goes_to_the_oldest_submissions_in_order():
    d = decide(
        state(
            waiting=waiting(OLDER, OLDEST, RECENT, ids=["mid", "old", "young"]),
            instances=[inst("w1"), inst("w2")],
            ready=["w1", "w2"],
        ),
        ASSIGN,
    )

    assert [sub for sub, _ in d.assign] == ["old", "mid"]
    assert len({i for _, i in d.assign}) == 2, "no instance may take two submissions"


def test_equal_ages_still_order_deterministically():
    """A decision that changes between ticks for no reason is undebuggable from a log."""
    first = decide(
        state(waiting=waiting(OLDEST, OLDEST, ids=["b", "a"]), instances=[inst("w1")],
              ready=["w1"]),
        ASSIGN,
    )
    again = decide(
        state(waiting=waiting(OLDEST, OLDEST, ids=["a", "b"]), instances=[inst("w1")],
              ready=["w1"]),
        ASSIGN,
    )

    assert first.assign == again.assign == (("a", "id-w1"),)


def test_an_instance_that_has_not_registered_is_never_assigned():
    """The stall that has no signal: a chain published to a queue with no consumer.

    Capacity is optimistic -- a booting instance will serve, so it must not provoke a
    second create -- while assignment is pessimistic. Both readings appear here: nothing
    is assigned, and nothing is created either, because the instance is expected to
    register shortly and take the work.
    """
    d = decide(state(waiting=waiting(OLDEST), instances=[inst("booting")]), ASSIGN)

    assert d.assign == ()
    assert len(d.create) == 0
    assert any("none assignable" in r for r in d.reasons)


def test_an_instance_that_already_holds_a_submission_is_not_assigned_another():
    """`spent` means "has been given work" under S2, and one VM serves one submission."""
    d = decide(
        state(
            waiting=waiting(OLDEST),
            instances=[inst("w1")],
            ready=["w1"],
            spent=["w1"],
        ),
        ASSIGN,
    )

    assert d.assign == ()
    assert len(d.create) == 1, "a spent instance is not capacity for a waiting submission"


def test_an_instance_being_reaped_this_round_is_not_assigned():
    """Publishing to a VM this same tick deletes loses the chain outright."""
    limits = Limits(
        max_instances=5, assign=True, max_lifetime=timedelta(minutes=3),
        assign_max_age=timedelta(minutes=30),
    )
    doomed = inst("doomed", age=timedelta(minutes=5))
    d = decide(state(waiting=waiting(OLDEST), instances=[doomed], ready=["doomed"]),
               limits)

    assert d.delete == ("id-doomed",)
    assert d.assign == ()


def test_an_instance_past_the_assign_ceiling_is_neither_assigned_nor_capacity():
    """The poweroff race, and the deadlock a half-fix would cause.

    The worker's supervisor may power the VM off from ``BOOT_GRACE_SEC`` (600 s uptime)
    onwards, so at nine minutes a chain published to it can lose its consumer seconds
    later. Excluding it from assignment *alone* would leave it counted as capacity --
    demand satisfied, work unassignable, nothing created -- which is the run-3/run-5
    stall reached through a new door. Both halves, or neither.
    """
    stale = inst("stale", age=timedelta(minutes=9))
    d = decide(state(waiting=waiting(RECENT), instances=[stale], ready=["stale"]), ASSIGN)

    assert d.assign == ()
    assert len(d.create) == 1
    assert any("too old to assign" in a for a in d.alerts)


def test_ageing_out_with_nothing_waiting_is_not_worth_a_warning():
    """Routine: an over-provisioned instance idling towards its own poweroff.

    Alerting here would fire on the tail of every submission that got two instances
    from the prefetch race, which is how an alert channel stops being read.
    """
    stale = inst("stale", age=timedelta(minutes=9))
    d = decide(state(instances=[stale], ready=["stale"]), ASSIGN)

    assert d.assign == ()
    assert not d.alerts


def test_an_instance_inside_the_ceiling_is_assigned_normally():
    fresh = inst("fresh", age=timedelta(minutes=7))
    d = decide(state(waiting=waiting(RECENT), instances=[fresh], ready=["fresh"]), ASSIGN)

    assert d.assign == (("sub-0", "id-fresh"),)
    assert not d.alerts


def test_the_youngest_assignable_instance_is_preferred():
    """It is the furthest from its own poweroff clock; the older one is nearly spent."""
    d = decide(
        state(
            waiting=waiting(OLDEST),
            instances=[inst("old", age=timedelta(minutes=7)),
                       inst("new", age=timedelta(minutes=1))],
            ready=["old", "new"],
        ),
        ASSIGN,
    )

    assert d.assign == (("sub-0", "id-new"),)


def test_an_undated_instance_is_assigned_last():
    """Its age cannot be checked against the ceiling, so it is not the first choice."""
    undated = Instance(id="id-undated", name="undated", status="ACTIVE", created_at=None)
    d = decide(
        state(
            waiting=waiting(OLDEST),
            instances=[undated, inst("dated", age=timedelta(minutes=2))],
            ready=["undated", "dated"],
        ),
        ASSIGN,
    )

    assert d.assign == (("sub-0", "id-dated"),)


def test_a_fresh_submission_is_demand_immediately_when_armed():
    """The two-minute grace exists for a race S2 removes; keeping it would add latency.

    Off, a submission unclaimed for three seconds is the ordinary gap between celery
    handing over the head task and ``prepare_submission`` claiming it, and counting it
    would create an instance for a submission already starting. Armed, nothing is
    published until an instance exists, so there is no such gap and the same reading is
    simply a submission with nowhere to run.
    """
    fresh = state(waiting=waiting(FRESH))

    assert len(decide(fresh, LIMITS).create) == 0
    assert len(decide(fresh, ASSIGN).create) == 1


def test_a_waiting_submission_at_depth_zero_is_not_an_anomaly_when_armed():
    """It is the design. Alerting on it would fire on every tick of a healthy fleet."""
    d = decide(state(depth=0, waiting=waiting(OLDEST)), ASSIGN)

    assert not d.alerts
    assert len(d.create) == 1


def test_a_shared_queue_message_is_not_provisioned_for_twice():
    """Rollout step 3 runs both paths: a dispatched submission is also unassigned."""
    d = decide(state(depth=1, waiting=waiting(OLDEST)), ASSIGN)

    assert len(d.create) == 1, "max(depth, waiting), never depth + waiting"


def test_assignment_and_creation_are_decided_together():
    """One tick: place what can be placed, create for the remainder, count once."""
    d = decide(
        state(waiting=waiting(OLDEST, OLDER, RECENT), instances=[inst("w1")],
              ready=["w1"]),
        ASSIGN,
    )

    assert len(d.assign) == 1
    assert len(d.create) == 2, "three waiting, one available: two more instances"


def test_the_breaker_does_not_block_assignment():
    """It means "stop spending money"; placing work on a paid-for instance is not that.

    Gating assignment on the breaker would strand submissions on idle instances that
    are already running, then let those instances power themselves off unused -- paying
    for the fleet and getting nothing from it.
    """
    d = decide(
        state(waiting=waiting(OLDEST), instances=[inst("w1")], ready=["w1"], failures=9),
        ASSIGN,
    )

    assert len(d.create) == 0
    assert d.assign == (("sub-0", "id-w1"),)


def test_nothing_waiting_assigns_nothing():
    d = decide(state(instances=[inst("w1")], ready=["w1"]), ASSIGN)

    assert d.assign == ()


def test_every_assignment_is_explained():
    """Assignments are logged verbatim; a binding with no reason is unauditable."""
    d = decide(
        state(waiting=waiting(OLDEST, OLDER), instances=[inst("w1"), inst("w2")],
              ready=["w1", "w2"]),
        ASSIGN,
    )

    for sub, _ in d.assign:
        assert any(sub in r for r in d.reasons)


# --- the loop, not the tick -------------------------------------------------
#
# Every case above checks one call. The property P2's exit criterion actually asks for
# is over many: N submissions from an empty fleet, each placed on its own instance,
# none stranded, none placed twice, and no runaway creation. One tick being right does
# not imply the sequence converges -- an assigner that forgot `spent` would place the
# same submission on a new instance every 30 s and look correct in isolation.

TICK = timedelta(seconds=30)
#: Boot to readiness marker, measured across five production runs (D9).
BOOT_TO_READY = timedelta(seconds=140)


def _simulate(n_submissions, limits=ASSIGN, ticks=40):
    """Run `decide` in a loop against a fleet that boots, registers and stays put.

    Deliberately does not simulate a worker *finishing*: the question here is placement,
    and a submission that completes would free capacity and hide a stranding bug.
    """
    now = NOW
    created, ready, placed, creates = {}, set(), {}, 0
    submitted = {f"sub-{i}": now for i in range(n_submissions)}

    for tick in range(ticks):
        ready |= {i for i, at in created.items() if now - at >= BOOT_TO_READY}
        d = decide(
            FleetState(
                queue_depth=0,
                serving=len(placed),
                instances=tuple(
                    Instance(id=i, name=i, status="ACTIVE", created_at=at)
                    for i, at in created.items()
                ),
                # `spent` is what the assigner's own claim writes, so an instance it
                # has placed work on is spent from the next tick onwards.
                spent=frozenset(placed.values()),
                ready=frozenset(ready),
                waiting=tuple(
                    WaitingSubmission(id=s, age=now - at)
                    for s, at in submitted.items()
                    if s not in placed
                ),
                now=now,
            ),
            limits,
        )
        for sub, inst_id in d.assign:
            assert sub not in placed, f"{sub} placed twice"
            assert inst_id not in placed.values(), f"{inst_id} given two submissions"
            placed[sub] = inst_id
        for _rung in d.create:
            creates += 1
            created[f"i-{creates}"] = now
        for inst_id in d.delete:
            created.pop(inst_id, None)
        if len(placed) == n_submissions:
            return tick, creates, placed
        now += TICK
    return None, creates, placed


def test_a_fleet_of_submissions_converges_one_instance_each():
    tick, creates, placed = _simulate(5)

    assert len(placed) == 5
    assert creates == 5, "one instance per submission, and not one more"
    assert len(set(placed.values())) == 5, "no instance took two"
    # 140 s to readiness plus a tick to notice it: five ticks, and every submission
    # placed in the same round because they all became assignable together.
    assert tick == 5


def test_the_cap_makes_submissions_wait_rather_than_overbooking():
    """Eight submissions against max_instances=5: five placed, three wait, five created.

    The failure this pins is the tempting one -- an assigner that recomputed demand
    without counting what it had already placed would create past the cap every tick
    and take an OpenStack quota rejection.
    """
    tick, creates, placed = _simulate(8)

    assert tick is None, "three submissions cannot be placed until a worker finishes"
    assert creates == 5
    assert len(placed) == 5


# --- mixed mode: demand and assignability part company (rollout step 3) ------


def test_a_submission_girder_already_dispatched_is_never_placed():
    """Two workers on one workspace, arriving through the operator's door.

    Rollout step 3 runs both paths at once. A submission ``submit_job`` published to
    the shared queue is RUNNING and unclaimed, so it looks exactly like one waiting to
    be placed -- and publishing a second chain for it is the failure targeted
    assignment exists to remove.
    """
    d = decide(
        state(
            waiting=waiting(OLDEST, assignable=False),
            instances=[inst("w1")],
            ready=["w1"],
        ),
        ASSIGN,
    )

    assert d.assign == ()


def test_a_submission_girder_dispatched_still_counts_as_demand():
    """It needs an instance either way; only *who publishes* differs.

    Dropping it from demand would stop the fleet scaling for exactly the submissions
    that are mid-rollout, which is the stall class this signal was added to end.
    """
    d = decide(state(waiting=waiting(OLDEST, assignable=False)), ASSIGN)

    assert len(d.create) == 1


def test_the_placeable_head_of_the_line_is_served_past_an_unplaceable_one():
    """S7 is oldest-first among the submissions that are ours, not among all of them.

    An older submission on the other dispatch path is not a head-of-line block: a
    worker is already coming for it, so waiting behind it would be waiting for nothing.
    """
    d = decide(
        state(
            waiting=(
                WaitingSubmission(id="legacy", age=OLDEST, assignable=False),
                WaitingSubmission(id="ours", age=RECENT),
            ),
            instances=[inst("w1")],
            ready=["w1"],
        ),
        ASSIGN,
    )

    assert d.assign == (("ours", "id-w1"),)


def test_nothing_of_ours_to_place_does_not_read_as_a_stall():
    """The arming failure and the mixed-mode reading must not share a log line.

    "waiting, none assignable" is the line that says no worker announces readiness --
    the deadlock. A submission on the other path is not that, and a reader who cannot
    tell them apart will chase the wrong one.
    """
    d = decide(
        state(waiting=waiting(OLDEST, assignable=False), instances=[inst("w1")],
              ready=["w1"]),
        ASSIGN,
    )

    assert any("none of them ours to place" in r for r in d.reasons)
    assert not any("none assignable" in r for r in d.reasons)


# ---------------------------------------------------------------------------
# P3: a heterogeneous fleet. Everything below needs BOTH a catalogue and
# `assign=True`; with either missing the arithmetic is the pre-P3 one, and the
# tests above are what assert that.
# ---------------------------------------------------------------------------

#: Two rungs from the real Jetstream2 ladder, verified against live Nova 2026-08-20
#: (m3.medium 8 vCPU / 30720 MiB, m3.large 16 vCPU / 61440 MiB). Which rung a given
#: deployment defaults to is its own configuration and not this function's business.
LADDER = (SizeSpec(memory_gb=30, vcpus=8), SizeSpec(memory_gb=60, vcpus=16))
SIZED = Limits(max_instances=5, assign=True, sizes=LADDER)


def test_an_empty_catalogue_is_the_pre_p3_arithmetic():
    """The equivalence the whole phase order rests on.

    Production is flag-off with no catalogue, so it must take the old path exactly.
    Same state, two limits differing only in `sizes`: identical decisions.
    """
    s = state(depth=0, instances=[inst("a")], ready=["a"], waiting=waiting(MIN * 3))
    bare = decide(s, Limits(max_instances=5, assign=True))
    assert len(bare.create) == 0
    assert bare.assign == (("sub-0", "id-a"),)
    # And with a catalogue, an unsized instance is the smallest rung and an unsized
    # submission wants the smallest rung, so they still meet.
    sized = decide(s, SIZED)
    assert sized.assign == bare.assign
    assert len(sized.create) == len(bare.create)


def test_an_instance_of_the_wrong_size_is_not_capacity():
    """Two 60 GB submissions against one idle 30 GB instance need TWO creates.

    The scalar formula says one -- `demand(2) - available(1)` -- and that is the exact
    assumption that stops being true once the fleet is heterogeneous. Getting this
    wrong strands the second submission behind capacity that can never serve it, which
    is the run-3/run-5 stall shape in a new dimension.
    """
    d = decide(
        state(
            instances=[inst("small", size=30)],
            ready=["small"],
            waiting=waiting(MIN * 4, MIN * 3, sizes=[60, 60]),
        ),
        SIZED,
    )
    assert rungs(d) == (60, 60)
    assert d.assign == (), "a 30 GB instance must never take a 60 GB submission"


def test_a_submission_is_assigned_only_to_its_own_size():
    d = decide(
        state(
            instances=[inst("small", size=30), inst("big", size=60)],
            ready=["small", "big"],
            waiting=waiting(MIN * 3, sizes=[60]),
        ),
        SIZED,
    )
    assert d.assign == (("sub-0", "id-big"),)


def test_assignment_stops_at_the_head_of_the_line_rather_than_skipping():
    """S7. The 60 GB submission is oldest and cannot be placed; the 30 GB one waits.

    Skipping would raise utilisation and starve the large sizes forever, because under
    a steady stream of small submissions the expensive one is always the one that does
    not fit -- and it is unexplainable to a researcher watching later submissions run.
    """
    d = decide(
        state(
            instances=[inst("small", size=30)],
            ready=["small"],
            waiting=waiting(MIN * 9, MIN * 2, sizes=[60, 30]),
        ),
        SIZED,
    )
    assert d.assign == (), "nothing behind the blocked head may be assigned"
    assert any("head of line" in r for r in d.reasons)
    assert any("head of line" in a for a in d.alerts), "a blocked head is an anomaly"
    # ...and the fleet still fixes it: an instance of the shape it wants is created.
    assert 60 in rungs(d)


def test_the_vcpu_quota_binds_before_the_instance_count():
    """S6, superseding D3. 40 vCPU of quota is five 8-vCPU instances, not twenty-five.

    D3's rule -- count instances, not vCPUs -- holds only for a uniform fleet. With the
    instance cap raised out of the way the vCPU quota is what must bind, or the
    controller takes a Nova rejection instead, which is the failure D3 wrote its rule
    to avoid.
    """
    limits = Limits(max_instances=25, assign=True, sizes=LADDER, max_vcpus=40)
    d = decide(state(waiting=waiting(*[MIN * 3] * 10, sizes=[30] * 10)), limits)
    assert len(d.create) == 5, "40 vCPU / 8 per instance"
    assert any("head of line" in a for a in d.alerts)


def test_the_ram_quota_binds_at_the_larger_rung():
    """Nine 125 GB instances exhaust 1220 GB, which is S6's worked example."""
    ladder = (SizeSpec(memory_gb=125, vcpus=32),)
    limits = Limits(
        max_instances=25, assign=True, sizes=ladder, max_vcpus=320, max_ram_gb=1220
    )
    d = decide(state(waiting=waiting(*[MIN * 3] * 12, sizes=[125] * 12)), limits)
    assert len(d.create) == 9, "1220 GB / 125 per instance"


def test_quota_counts_what_is_already_live():
    limits = Limits(max_instances=25, assign=True, sizes=LADDER, max_vcpus=40)
    d = decide(
        state(
            instances=[inst("a", size=30), inst("b", size=30)],
            spent=["a", "b"],
            waiting=waiting(*[MIN * 3] * 10, sizes=[30] * 10),
        ),
        limits,
    )
    assert len(d.create) == 3, "two live instances already hold 16 of the 40 vCPU"


def test_a_create_that_will_not_fit_stops_the_ones_behind_it():
    """S7 again, on the create path: strictly oldest-first, no skipping.

    40 vCPU of quota. Two 60 GB instances (16 vCPU each) fit, the third does not -- and
    the 30 GB submission behind it *would* fit in the remaining 8 vCPU. It must still
    not be created: that is the difference between stopping and skipping, and the
    quota has to be sized so the two answers differ or this test pins nothing. (It
    originally used 32 vCPU, where the trailing 30 did not fit either and `continue`
    passed the suite unchanged -- caught by mutation, not by review.)
    """
    limits = Limits(max_instances=25, assign=True, sizes=LADDER, max_vcpus=40)
    d = decide(
        state(
            waiting=waiting(
                MIN * 9, MIN * 8, MIN * 7, MIN * 2, sizes=[60, 60, 60, 30]
            )
        ),
        limits,
    )
    assert rungs(d) == (60, 60)
    assert any("does not fit the quota" in a for a in d.alerts)


def test_a_size_that_left_the_catalogue_is_visible_not_silently_downgraded():
    """S1 calls withdrawing a rung a compatibility event; this is what it looks like.

    The dangerous handling is rounding 125 down to 60 and running the analysis on
    hardware it was never sized for. It blocks instead, loudly, and ages out through
    `sivacor.assignment_timeout` as `reaped_no_worker` -- which points at the fleet.
    """
    d = decide(state(waiting=waiting(MIN * 3, sizes=[125])), SIZED)
    assert rungs(d) == ()
    assert any("not in the catalogue" in a for a in d.alerts)


def test_an_unsized_submission_gets_the_cheapest_rung():
    """Pre-P1 submissions have no `requested_memory_gb`, and guessing up costs money."""
    d = decide(state(waiting=waiting(MIN * 3)), SIZED)
    assert rungs(d) == (30,)


def test_an_untagged_instance_counts_as_the_cheapest_rung():
    """Pre-P3 instances have no size tag. Under-counting cannot stall anything; Nova
    refusing the create is backpressure `QuotaExceeded` already handles (S6)."""
    d = decide(
        state(
            instances=[inst("legacy")],
            ready=["legacy"],
            waiting=waiting(MIN * 3, sizes=[30]),
        ),
        SIZED,
    )
    assert d.assign == (("sub-0", "id-legacy"),), "matched to the smallest rung"
    assert rungs(d) == ()


def test_unarmed_keeps_the_scalar_arithmetic_even_with_a_catalogue():
    """The guard that lets P3 land while production still runs the shared queue.

    Unarmed, a 30 GB instance IS capacity for a 60 GB submission, because on the shared
    queue any worker consumes any message -- size gates nothing. So the scalar reading
    is the correct one there, and the per-size path must not run: it would see a
    shortfall of one and boot an instance nobody needs, on the deployment least able to
    afford a surprise.

    This is the test mutation testing said was missing: dropping `limits.assign` from
    that condition passed the whole suite.
    """
    s = state(
        depth=1,
        instances=[inst("small", size=30)],
        waiting=waiting(timedelta(minutes=5), sizes=[60]),
    )
    unarmed = decide(s, Limits(max_instances=5, sizes=LADDER))
    assert rungs(unarmed) == (), "an available worker will take it off the shared queue"
    assert unarmed.assign == (), "unarmed, this controller places nothing"

    # Armed, the same state is a real shortfall: nothing will hand that message to the
    # 30 GB box, and the 60 GB submission needs hardware that does not exist yet.
    assert rungs(decide(s, replace(SIZED, max_instances=5))) == (60,)


def test_unarmed_creates_are_unsized():
    """Demand from queue depth carries no size, so the caller boots its own default."""
    d = decide(state(depth=2), Limits(max_instances=5, sizes=LADDER))
    assert rungs(d) == (None, None)


def test_a_quota_stop_does_not_blame_the_instance_cap():
    """The mirror's log, 2026-08-20, reduced to an assertion.

    Two workers left over from an earlier test (8 + 16 = 24 vCPU) against a max_vcpus of
    8, with instance slots to spare. The fleet is genuinely over quota, so nothing can be
    created -- but the old code reported it as `CAPPED: want 2 more but only 3 slot(s)
    left of max_instances=5`, which reads as "raise max_instances" when raising it would
    change nothing.

    Counting spent-but-live instances is correct here and not the bug: a spent instance
    really does hold its vCPU against the allocation until it is reaped.
    """
    limits = Limits(max_instances=5, assign=True, sizes=LADDER, max_vcpus=8)
    d = decide(
        state(
            instances=[inst("small", size=30), inst("big", size=60)],
            spent=["small", "big"],
            waiting=waiting(MIN * 4, MIN * 3, sizes=[30, 30]),
        ),
        limits,
    )

    assert rungs(d) == (), "24 vCPU are already held; nothing fits"
    assert any("does not fit the quota" in a for a in d.alerts)
    assert any("vCPU 24+8 > 8" in a for a in d.alerts), "name the real arithmetic"
    # The point of the fix:
    assert not any("CAPPED" in r for r in d.reasons), "the instance cap did not bind"
    assert any("instance cap is not what bound" in r for r in d.reasons)


def test_the_instance_cap_still_says_CAPPED_when_it_is_what_bound():
    """The other half: don't lose the message that has been there since D3."""
    limits = Limits(max_instances=2, assign=True, sizes=LADDER)
    d = decide(
        state(waiting=waiting(*[MIN * 5] * 5, sizes=[30] * 5)), limits
    )
    assert len(d.create) == 2
    assert any("CAPPED" in r and "max_instances=2" in r for r in d.reasons)
