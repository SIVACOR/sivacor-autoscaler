"""Controller-level state that :mod:`plan` cannot hold, because it is not a function.

Both cases here are regressions of a stall that has now been produced four distinct
ways (plan runs 3, 5, 6 and the 2026-08-05 production run): a submission waiting while
the controller believes an instance is available. The arithmetic in :mod:`plan` is
correct in every one of them; what was wrong was ``spent``.
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sivacor_autoscaler import controller as controller_mod
from sivacor_autoscaler.controller import Config, Controller
from sivacor_autoscaler.plan import Decision, FleetState, Instance

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

    def create_instance(_conn, _cfg, _user_data):
        conn.creates += 1
        if conn.fail:
            raise RuntimeError("user_data is 65600 bytes encoded, over Nova's 65535")
        return "new-id"

    monkeypatch.setattr(controller_mod.fleet, "create_instance", create_instance)
    # decide() is not under test here; drive the create loop directly.
    monkeypatch.setattr(controller_mod, "decide", lambda state, limits: Decision(create=1))
    monkeypatch.setattr(ctl, "gather", lambda: FleetState(queue_depth=1, serving=0, now=NOW))


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
