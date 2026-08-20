"""Controller-level state that :mod:`plan` cannot hold, because it is not a function.

Both cases here are regressions of a stall that has now been produced four distinct
ways (plan runs 3, 5, 6 and the 2026-08-05 production run): a submission waiting while
the controller believes an instance is available. The arithmetic in :mod:`plan` is
correct in every one of them; what was wrong was ``spent``.
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sivacor_autoscaler import controller as controller_mod
from sivacor_autoscaler.controller import Config, Controller
from sivacor_autoscaler.plan import Decision, FleetState, Instance, Limits

NOW = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)


class FakeCursor:
    def __init__(self, docs):
        self._docs = docs

    def sort(self, *a):
        return self

    def limit(self, *a):
        return self

    def __iter__(self):
        return iter(self._docs)


class FakeGirder:
    """``db[collection]`` with a mutable document list, so a test can delete a job."""

    def __init__(self, jobs):
        self.jobs = jobs

    def __getitem__(self, name):
        return self

    def find(self, query, projection=None):
        return FakeCursor(self.jobs)

    def count_documents(self, query):
        return 0

    def find_one(self, query):
        # The arm flag, read every tick from Girder's setting document. Absent means
        # off, which is what every test here assumes unless it says otherwise.
        return None


class FakeRedis:
    def llen(self, queue):
        return 0

    def keys(self, pattern):
        return []


def claim(instance_id):
    return {"_id": instance_id, "meta": {"worker_queue": f"sivacor.{instance_id}"}}


def controller(jobs):
    cfg = Config(
        template=Path("/nonexistent"),
        manager_ip="10.0.0.1",
        master_key_hex="ab",
        redis_password="pw",
        deployment="test.sivacor.org",
    )
    return Controller(object(), FakeRedis(), FakeGirder(jobs), cfg)


def live(instance_id, age_minutes=5):
    return Instance(
        id=instance_id,
        name=f"sivacor-worker-{instance_id}",
        status="ACTIVE",
        created_at=NOW - timedelta(minutes=age_minutes),
    )


def test_a_claim_survives_the_submission_being_deleted():
    """The 2026-08-05 production regression, in one assertion.

    ``DELETE /sivacor/submission/:id`` calls ``Job().remove()``, so the only record
    that an instance claimed anything can be erased by a user while that instance is
    still winding down. It then reads as available capacity and the next submission
    stalls behind it for the rest of the worker's idle tail -- measured at 12 min 36 s.
    """
    jobs = [claim("uuid-a")]
    ctl = controller(jobs)

    assert ctl._spent([live("uuid-a")]) == frozenset({"uuid-a"})

    jobs.clear()  # the researcher deletes their completed submission

    assert ctl._spent([live("uuid-a")]) == frozenset({"uuid-a"}), (
        "a claim is irreversible; forgetting it re-offers a spent worker as capacity"
    )


def test_a_reaped_instance_is_forgotten():
    """Retention is bounded by the fleet, or the cache grows for the pilot's lifetime."""
    ctl = controller([claim("uuid-a")])
    ctl._spent([live("uuid-a")])

    assert ctl._spent([]) == frozenset()
    assert ctl._spent_seen == set()


def test_the_static_worker_never_enters_the_cache():
    """`sivacor.static-01` carries the prefix but names no instance (see test_signals)."""
    ctl = controller([claim("static-01"), claim("uuid-a")])

    assert ctl._spent([live("uuid-a")]) == frozenset({"uuid-a"})


def test_claims_accumulate_across_ticks():
    """Two workers claiming on different ticks must both stay spent."""
    jobs = [claim("uuid-a")]
    ctl = controller(jobs)
    ctl._spent([live("uuid-a"), live("uuid-b")])

    jobs.append(claim("uuid-b"))

    assert ctl._spent([live("uuid-a"), live("uuid-b")]) == frozenset(
        {"uuid-a", "uuid-b"}
    )


# --- the circuit breaker must be able to close again (2026-08-12) ------------
# `consecutive_failures` was only ever zeroed in __init__. plan.decide forces
# create=0 at the threshold, so the breaker prevented the very success that would
# reset it: it latched until the process was restarted. An oversized user_data tripped
# it in ~90 s and the fleet made no workers for five minutes, through the fix being
# deployed, logging only the routine "BREAKER OPEN" line.


class _CreateConn:
    """A controller wired so creates can be made to fail or succeed on demand."""

    def __init__(self):
        self.creates = 0
        self.fail = False


def _armed(monkeypatch, ctl, conn):
    """Point the controller's create path at `conn`, bypassing OpenStack entirely."""
    monkeypatch.setattr(controller_mod.fleet, "build_user_data", lambda *a, **k: "#!/bin/bash\n")

    def create_instance(_conn, _cfg, _user_data, flavor=None, size=None):
        conn.creates += 1
        if conn.fail:
            raise RuntimeError("user_data is 65600 bytes encoded, over Nova's 65535")
        return "new-id"

    monkeypatch.setattr(controller_mod.fleet, "create_instance", create_instance)
    # decide() is not under test here; drive the create loop directly.
    monkeypatch.setattr(controller_mod, "decide", lambda state, limits: Decision(create=(None,)))
    monkeypatch.setattr(
        ctl, "gather", lambda limits=None: FleetState(queue_depth=1, serving=0, now=NOW)
    )


def test_a_successful_create_clears_the_breaker(monkeypatch):
    ctl, conn = controller([]), _CreateConn()
    _armed(monkeypatch, ctl, conn)

    conn.fail = True
    ctl.step()
    assert ctl.consecutive_failures == 1

    conn.fail = False
    ctl.step()
    assert ctl.consecutive_failures == 0, "a create that works is evidence the fault is over"
    assert ctl._last_failure is None


def test_the_breaker_reopens_after_the_cooldown(monkeypatch):
    """THE latch. Resetting on success is necessary and not sufficient.

    Once decide() refuses to create, no attempt happens, so no success can occur, so
    nothing clears the counter. Only elapsed time can.
    """
    ctl, conn = controller([]), _CreateConn()
    _armed(monkeypatch, ctl, conn)
    ctl.consecutive_failures = 3
    ctl._last_failure = time.monotonic()

    ctl._expire_breaker()
    assert ctl.consecutive_failures == 3, "still inside the cooldown: stay open"

    ctl._last_failure = time.monotonic() - ctl.cfg.breaker_cooldown - 1
    ctl._expire_breaker()
    assert ctl.consecutive_failures == 0, "cooldown elapsed: allow one attempt again"


def test_reopening_is_one_attempt_not_a_return_to_trust(monkeypatch):
    """If the retry fails, the fleet waits another cooldown rather than looping."""
    ctl, conn = controller([]), _CreateConn()
    _armed(monkeypatch, ctl, conn)
    ctl.consecutive_failures = 3
    ctl._last_failure = time.monotonic() - ctl.cfg.breaker_cooldown - 1
    conn.fail = True

    ctl.step()

    assert conn.creates == 1, "exactly one probe attempt"
    assert ctl.consecutive_failures == 1
    assert ctl._last_failure is not None, "the clock restarts, so the next wait is a full cooldown"


def test_a_quiet_controller_never_reopens_anything(monkeypatch):
    """No failures recorded means no cooldown bookkeeping and no spurious warning."""
    ctl = controller([])
    ctl._expire_breaker()
    assert ctl.consecutive_failures == 0 and ctl._last_failure is None


# --- what gather() must read once assignment is armed ----------------------


class CountingRedis(FakeRedis):
    """Records whether the readiness markers were read at all."""

    def __init__(self):
        self.key_patterns = []

    def keys(self, pattern):
        self.key_patterns.append(pattern)
        return []


class EmptyCloud:
    """An OpenStack connection with no servers in it; gather() lists the fleet first."""

    class compute:
        @staticmethod
        def servers(details=True):
            return []


def _controller_with(redis, limits):
    cfg = Config(
        template=Path("/nonexistent"),
        manager_ip="10.0.0.1",
        master_key_hex="ab",
        redis_password="pw",
        deployment="test.sivacor.org",
        limits=limits,
    )
    return Controller(EmptyCloud(), redis, FakeGirder([]), cfg)


def test_readiness_is_read_when_assignment_is_armed():
    """Deadlock-shaped if this regresses, and invisible from every other number.

    An instance that has not registered with the broker is never assigned to, so an
    empty ``ready`` set means nothing is ever assignable -- while depth, live, spent and
    serving all read exactly as they do on a healthy idle fleet. The production
    ``provision_deadline`` is ``None``, so the pre-existing condition alone would have
    left this signal unread precisely when it became load-bearing.
    """
    redis = CountingRedis()
    _controller_with(redis, Limits(assign=True)).gather()

    assert redis.key_patterns == ["sivacor:ready:*"]


def test_readiness_is_left_unread_when_nothing_will_consult_it():
    """gather() is all-or-nothing, so an unused signal must not be able to fail a round."""
    redis = CountingRedis()
    state = _controller_with(redis, Limits()).gather()

    assert redis.key_patterns == []
    assert state.ready == frozenset()


# --- executing an assignment (P2) -------------------------------------------


class ArmedGirder(FakeGirder):
    """A database whose arm flag is on, and which records claims."""

    def __init__(self, jobs, armed=True):
        super().__init__(jobs)
        self.armed = armed

    def find_one(self, query):
        return {"key": "sivacor.targeted_assignment", "value": self.armed}


def _assigning(monkeypatch, ctl, decision, calls):
    """Drive step() straight at one decision, with dispatch stubbed out."""
    monkeypatch.setattr(controller_mod, "decide", lambda state, limits: decision)
    monkeypatch.setattr(
        ctl, "gather", lambda limits=None: FleetState(queue_depth=0, serving=0, now=NOW)
    )
    monkeypatch.setattr(
        controller_mod.dispatch,
        "assign",
        lambda db, sub, inst, queue: calls.append((sub, inst, queue)) or True,
    )


def test_an_assignment_is_executed_against_the_instances_private_queue(monkeypatch):
    """The one line that is the feature, reached from a decision.

    Before this the executor did not exist and step() logged an ERROR instead -- a
    decision nothing acts on, which is how two changes in autoscaling_plan.md shipped
    inert for two loop tests.
    """
    ctl, calls = controller([]), []
    _assigning(monkeypatch, ctl, Decision(assign=(("sub-1", "uuid-a"),)), calls)

    ctl.step()

    assert calls == [("sub-1", "uuid-a", "sivacor")]


def test_one_bad_binding_does_not_strand_the_rest_of_the_round(monkeypatch):
    """A submission whose chain will not build is one submission, not an outage."""
    ctl, calls = controller([]), []
    _assigning(
        monkeypatch,
        ctl,
        Decision(assign=(("sub-1", "uuid-a"), ("sub-2", "uuid-b"))),
        calls,
    )

    def explode(db, sub, inst, queue):
        if sub == "sub-1":
            raise RuntimeError("chain would not build")
        calls.append((sub, inst, queue))
        return True

    monkeypatch.setattr(controller_mod.dispatch, "assign", explode)

    ctl.step()

    assert calls == [("sub-2", "uuid-b", "sivacor")]


def test_a_bad_binding_never_touches_the_circuit_breaker(monkeypatch):
    """The breaker counts instance creations, and it stops the WHOLE fleet at three.

    Feeding it submission-shaped failures would let one researcher's bad workflow
    take the fleet down for a cooldown -- the same shape as the 2026-08-12 latch.
    """
    ctl, calls = controller([]), []
    _assigning(monkeypatch, ctl, Decision(assign=(("sub-1", "uuid-a"),)), calls)
    monkeypatch.setattr(
        controller_mod.dispatch,
        "assign",
        lambda *a: (_ for _ in ()).throw(RuntimeError("nope")),
    )

    for _ in range(4):
        ctl.step()

    assert ctl.consecutive_failures == 0


def test_the_arm_flag_comes_from_girder_and_reaches_decide(monkeypatch):
    """One value, two processes: this controller must not have an opinion of its own.

    Any state where Girder's flag and this one disagree is broken -- both publishing
    is two workers on one workspace, neither is a fleet that looks healthy and idle
    while every submission waits.
    """
    seen = []
    ctl = Controller(EmptyCloud(), FakeRedis(), ArmedGirder([], armed=True), controller([]).cfg)
    monkeypatch.setattr(
        controller_mod,
        "decide",
        lambda state, limits: seen.append(limits.assign) or Decision(),
    )
    monkeypatch.setattr(
        ctl, "gather", lambda limits=None: FleetState(queue_depth=0, serving=0, now=NOW)
    )

    ctl.step()
    ctl.db.armed = False
    ctl.step()

    assert seen == [True, False], "flipping the setting takes effect without a restart"


def test_a_round_is_skipped_rather_than_guessing_the_arm_flag(monkeypatch):
    """Both defaults are wrong in the deployment where it matters, so neither is taken."""
    class BrokenSettings(FakeGirder):
        def find_one(self, query):
            raise RuntimeError("mongo down")

    ctl = Controller(EmptyCloud(), FakeRedis(), BrokenSettings([]), controller([]).cfg)
    monkeypatch.setattr(
        controller_mod,
        "decide",
        lambda state, limits: pytest.fail("decided on a guessed arm flag"),
    )

    ctl.step()  # no exception, no decision


def test_the_arm_flag_line_says_state_at_startup_and_change_on_a_flip(monkeypatch, caplog):
    """This line is what an operator greps to confirm a fresh deployment.

    The first reading is a statement of state, not a transition: "now OFF ... no
    longer" on startup reads as though somebody had just disarmed it. Observed on the
    mirror the first time the line was ever used, 2026-08-19.
    """
    ctl = Controller(
        EmptyCloud(), FakeRedis(), ArmedGirder([], armed=False), controller([]).cfg
    )

    with caplog.at_level("WARNING"):
        ctl.limits()
        assert "targeted assignment is OFF" in caplog.text
        assert "is now" not in caplog.text

        caplog.clear()
        ctl.db.armed = True
        ctl.limits()
        assert "targeted assignment is now ON" in caplog.text

        # Steady state is silent: this fires on change, not every 30 s.
        caplog.clear()
        ctl.limits()
        assert caplog.text == ""


# --- the catalogue reaches the arithmetic (P3.2) ----------------------------


def test_the_catalogue_reaches_decide_as_limits_sizes():
    """The seam. `plan` cannot read Girder and `catalogue` cannot do arithmetic, so if
    this hand-off is wrong the per-size path silently never runs and the fleet quietly
    boots one shape for everything -- which is exactly the pre-P3 bug P3 exists to fix.
    """
    from sivacor_autoscaler import catalogue

    ctl = controller([])
    ctl.rungs = (catalogue.Rung(30, "m3.medium", 8), catalogue.Rung(60, "m3.large", 16))

    limits = ctl.limits()

    assert [s.memory_gb for s in limits.sizes] == [30, 60]
    assert [s.vcpus for s in limits.sizes] == [8, 16]


def test_no_catalogue_means_no_sizes_which_is_the_pre_p3_path():
    assert controller([]).limits().sizes == ()


def test_the_flavour_map_lets_an_untagged_instance_be_placed():
    from sivacor_autoscaler import catalogue

    ctl = controller([])
    ctl.rungs = (catalogue.Rung(30, "m3.medium", 8), catalogue.Rung(60, "m3.large", 16))

    assert ctl._flavor_sizes() == {"m3.medium": 30, "m3.large": 60}
