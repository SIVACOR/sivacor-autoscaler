"""Tests for the scaling arithmetic and its guardrails.

This is the component whose bugs cost money in one direction and deadlock
submissions in the other, so the cases below are the ones worth being sure about
rather than a sweep for coverage.
"""

from datetime import datetime, timedelta, timezone

from sivacor_autoscaler.plan import (
    Decision,
    FleetState,
    Instance,
    Limits,
    WaitingSubmission,
    decide,
)

NOW = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
LIMITS = Limits(max_instances=5, max_lifetime=timedelta(hours=30), breaker_threshold=3)


def inst(name, status="ACTIVE", age=timedelta(minutes=5)):
    return Instance(id=f"id-{name}", name=name, status=status, created_at=NOW - age)


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


def waiting(*ages, ids=None):
    """Waiting submissions, named ``sub-0``, ``sub-1``, ... unless ``ids`` says otherwise."""
    names = ids or [f"sub-{i}" for i in range(len(ages))]
    return tuple(WaitingSubmission(id=n, age=a) for n, a in zip(names, ages))


#: Limits with the D9 check armed. The production default is None (off), so every
#: test that wants the behaviour has to ask for it explicitly -- which is the point.
D9 = Limits(provision_deadline=timedelta(minutes=10))


def test_scale_from_zero():
    d = decide(state(depth=3), LIMITS)
    assert d.create == 3


def test_nothing_queued_nothing_created():
    d = decide(state(depth=0, instances=[inst("w1")]), LIMITS)
    assert d == Decision(create=0, delete=(), reasons=d.reasons)
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

    assert d.create == 1, "a queued submission with only spent workers must scale up"


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

    assert d.create == 1, "a spent-but-alive worker must not count as capacity"
    assert any("1 spent" in r for r in d.reasons)


def test_available_instances_are_not_double_counted():
    """The other direction: an idle worker that has claimed nothing will take the work."""
    d = decide(state(depth=1, instances=[inst("w1")]), LIMITS)

    assert d.create == 0
    assert any("1 available" in r for r in d.reasons)


def test_booting_instances_are_not_double_counted():
    """An instance still in BUILD will take a queued submission; don't create twice."""
    d = decide(
        state(depth=2, serving=0, instances=[inst("w1", status="BUILD")]), LIMITS
    )
    assert d.create == 1


def test_cap_is_respected_and_says_so():
    """A cap that throttles silently is indistinguishable from a broken controller."""
    live = [inst(f"w{i}") for i in range(5)]
    spent = [f"w{i}" for i in range(5)]
    d = decide(state(depth=4, serving=5, instances=live, spent=spent), LIMITS)

    assert d.create == 0
    assert any("CAPPED" in r for r in d.reasons)
    assert any("will wait" in r for r in d.reasons)


def test_partial_cap_creates_what_it_can():
    live = [inst(f"w{i}") for i in range(4)]
    spent = [f"w{i}" for i in range(4)]
    d = decide(state(depth=3, serving=4, instances=live, spent=spent), LIMITS)
    assert d.create == 1  # one slot left of five


def test_shutoff_instances_are_reaped():
    d = decide(state(instances=[inst("done", status="SHUTOFF")]), LIMITS)
    assert d.delete == ("id-done",)
    assert any("SHUTOFF" in r for r in d.reasons)


def test_shutoff_instances_do_not_count_against_the_cap():
    """They hold no work; counting them would throttle the fleet as they accumulate."""
    dead = [inst(f"d{i}", status="SHUTOFF") for i in range(5)]
    d = decide(state(depth=2, instances=dead), LIMITS)
    assert d.create == 2
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
    assert d.create == 0
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
    assert d.create == 0
    assert d.delete == ("id-dead",)


def test_breaker_just_below_threshold_still_creates():
    d = decide(state(depth=1, failures=2), LIMITS)
    assert d.create == 1


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
    assert d.create == 1
    assert any("no readiness marker" in r for r in d.reasons)


def test_ready_instance_is_normal_capacity():
    ready = inst("ready", age=timedelta(minutes=30))
    d = decide(state(depth=1, instances=[ready], ready=["ready"]), D9)

    assert d.delete == ()
    assert d.create == 0


def test_instance_inside_the_deadline_is_left_alone():
    """Still booting is not the same as failed; boot->ready measured 122-140 s."""
    booting = inst("booting", age=timedelta(minutes=3))
    d = decide(state(depth=1, instances=[booting]), D9)

    assert d.delete == ()
    assert d.create == 0


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
    assert d.create == 1, (
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
    assert d.create == 0
    assert not d.alerts


def test_unclaimed_is_not_added_to_depth():
    """A queued submission is *also* unclaimed; counting both provisions twice."""
    d = decide(state(depth=1, unclaimed_ages=[STALE]), LIMITS)
    assert d.create == 1, "max(depth, unclaimed), never depth + unclaimed"
    assert not d.alerts, "depth already accounts for it, so nothing is anomalous"


def test_unclaimed_does_not_double_provision_against_available_capacity():
    d = decide(state(depth=0, instances=[inst("w1")], unclaimed_ages=[STALE]), LIMITS)
    assert d.create == 0, "an idle available worker will take it; no new instance"


def test_unclaimed_respects_the_breaker():
    d = decide(
        state(depth=0, unclaimed_ages=[STALE, STALE], failures=3),
        LIMITS,
    )
    assert d.create == 0, "a tripped breaker must not be bypassed by a new signal"


def test_unclaimed_absent_keeps_old_behaviour():
    d = decide(state(depth=2), LIMITS)
    assert d.create == 2
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
    assert d.create == 0
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
    assert d.create == 1, "a spent instance is not capacity for a waiting submission"


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
    assert d.create == 1
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

    assert decide(fresh, LIMITS).create == 0
    assert decide(fresh, ASSIGN).create == 1


def test_a_waiting_submission_at_depth_zero_is_not_an_anomaly_when_armed():
    """It is the design. Alerting on it would fire on every tick of a healthy fleet."""
    d = decide(state(depth=0, waiting=waiting(OLDEST)), ASSIGN)

    assert not d.alerts
    assert d.create == 1


def test_a_shared_queue_message_is_not_provisioned_for_twice():
    """Rollout step 3 runs both paths: a dispatched submission is also unassigned."""
    d = decide(state(depth=1, waiting=waiting(OLDEST)), ASSIGN)

    assert d.create == 1, "max(depth, waiting), never depth + waiting"


def test_assignment_and_creation_are_decided_together():
    """One tick: place what can be placed, create for the remainder, count once."""
    d = decide(
        state(waiting=waiting(OLDEST, OLDER, RECENT), instances=[inst("w1")],
              ready=["w1"]),
        ASSIGN,
    )

    assert len(d.assign) == 1
    assert d.create == 2, "three waiting, one available: two more instances"


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

    assert d.create == 0
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
        for _ in range(d.create):
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
