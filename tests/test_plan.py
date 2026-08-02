"""Tests for the scaling arithmetic and its guardrails.

This is the component whose bugs cost money in one direction and deadlock
submissions in the other, so the cases below are the ones worth being sure about
rather than a sweep for coverage.
"""

from datetime import datetime, timedelta, timezone

from sivacor_autoscaler.plan import Decision, FleetState, Instance, Limits, decide

NOW = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
LIMITS = Limits(max_instances=5, max_lifetime=timedelta(hours=30), breaker_threshold=3)


def inst(name, status="ACTIVE", age=timedelta(minutes=5)):
    return Instance(id=f"id-{name}", name=name, status=status, created_at=NOW - age)


def state(depth=0, serving=0, instances=(), failures=0, spent=(), ready=()):
    return FleetState(
        queue_depth=depth,
        serving=serving,
        instances=tuple(instances),
        spent=frozenset(f"id-{n}" for n in spent),
        ready=frozenset(f"id-{n}" for n in ready),
        consecutive_failures=failures,
        now=NOW,
    )


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
    """
    naive = Instance(
        id="n",
        name="n",
        status="ACTIVE",
        created_at=datetime(2026, 8, 1, 11, 0, tzinfo=timezone.utc),
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
