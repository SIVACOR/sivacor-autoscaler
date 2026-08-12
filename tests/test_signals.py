"""Tests for the Girder-side signals.

These parse a marker written by another repo, so most cases are about being
unsurprised by what Girder's job documents actually contain.

The signals read MongoDB directly rather than the REST API. That is not only about
the API key: ``GET /job`` silently scopes to the authenticated caller, which made
both signals return ``[]`` for the controller's entire life and cost an 18 min stall
(plan run 5). A query has no hidden scoping, so that class of bug cannot recur --
which is why the tests that used to pin the endpoint string are gone rather than
ported. See :data:`~sivacor_autoscaler.signals.JOB_COLLECTION`.
"""

from datetime import datetime, timedelta, timezone

import pytest

from sivacor_autoscaler.signals import (
    READY_KEY_PREFIX,
    SUBMISSION_TYPE,
    ready_instance_ids,
    serving_count,
    spent_instance_ids,
    unclaimed_submission_ages,
)


class FakeCursor:
    """Just enough of pymongo's cursor to record the query and yield documents."""

    def __init__(self, docs, calls):
        self._docs, self._calls = docs, calls

    def sort(self, key, direction):
        self._calls.append(("sort", key, direction))
        return self

    def limit(self, n):
        self._calls.append(("limit", n))
        return self

    def __iter__(self):
        return iter(self._docs)


class FakeCollection:
    def __init__(self, docs, raises=False):
        self.docs, self.raises, self.calls = docs, raises, []

    def find(self, query, projection=None):
        if self.raises:
            raise RuntimeError("mongo down")
        self.calls.append(("find", query, projection))
        return FakeCursor(self.docs, self.calls)

    def count_documents(self, query):
        if self.raises:
            raise RuntimeError("mongo down")
        self.calls.append(("count", query))
        return len(self.docs)


class FakeGirder:
    """Stands in for the Girder database: ``db[collection]``."""

    def __init__(self, jobs, raises=False):
        self.collection = FakeCollection(jobs, raises)

    def __getitem__(self, name):
        return self.collection

    @property
    def calls(self):
        return self.collection.calls


def job(queue=None, **meta):
    if queue is not None:
        meta["worker_queue"] = queue
    return {"_id": "j", "meta": meta} if meta else {"_id": "j"}


def test_extracts_instance_ids_from_queue_names():
    """The queue is named sivacor.<instance-uuid>, so the marker names the instance."""
    client = FakeGirder([job("sivacor.uuid-a"), job("sivacor.uuid-b")])

    assert spent_instance_ids(client) == frozenset({"uuid-a", "uuid-b"})


def test_unclaimed_jobs_are_ignored():
    """A job that no worker has picked up yet has no marker."""
    client = FakeGirder([job(), job(queue=None, status="queued"), job("sivacor.x")])

    assert spent_instance_ids(client) == frozenset({"x"})


def test_missing_or_null_meta_does_not_raise():
    """Girder omits `meta` entirely on some jobs, and may return it as None."""
    client = FakeGirder([{"_id": "j"}, {"_id": "k", "meta": None}])

    assert spent_instance_ids(client) == frozenset()


def test_foreign_queue_names_are_skipped():
    """The manager's static worker is not a fleet instance and must not be counted."""
    client = FakeGirder([job("sivacor.static-01"), job("local"), job("sivacor.uuid-a")])
    result = spent_instance_ids(client)

    assert "uuid-a" in result
    assert "local" not in result
    # static-01 does carry the prefix, so it *is* extracted -- documented rather than
    # silently wrong. It never matches an OpenStack instance id, so it cannot mark a
    # fleet instance spent; it only ever adds a harmless unmatched entry.
    assert result == frozenset({"static-01", "uuid-a"})


def test_duplicate_claims_collapse():
    """A worker that took two submissions (the prefetch case) is still one instance."""
    client = FakeGirder([job("sivacor.uuid-a"), job("sivacor.uuid-a")])

    assert spent_instance_ids(client) == frozenset({"uuid-a"})


def test_query_is_bounded_scoped_and_newest_first():
    """Unbounded would grow forever; unsorted would scan the wrong end of history."""
    client = FakeGirder([])
    spent_instance_ids(client)
    kinds = {c[0]: c for c in client.calls}

    assert kinds["find"][1] == {"type": SUBMISSION_TYPE}
    assert kinds["sort"][1:] == ("created", -1), "oldest-first would miss live workers"
    assert kinds["limit"][1] > 0


def test_girder_failure_propagates():
    """Returning an empty set would read as 'every worker is available'."""
    with pytest.raises(RuntimeError):
        spent_instance_ids(FakeGirder([], raises=True))


# -- serving_count --------------------------------------------------------------


def test_serving_counts_running_submissions():
    client = FakeGirder([job(), job(), job()])

    assert serving_count(client) == 3


def test_serving_filters_to_running_submissions():
    """Terminal submissions are not capacity, so the query must do the filtering."""
    client = FakeGirder([])
    serving_count(client)
    _, query = client.calls[0]

    assert query == {"type": SUBMISSION_TYPE, "status": 2}


def test_serving_failure_propagates():
    """Returning 0 would read as 'nothing is running' and could provoke a burst."""
    with pytest.raises(RuntimeError):
        serving_count(FakeGirder([], raises=True))


# -- ready_instance_ids (D9) -----------------------------------------------------


class FakeRedis:
    def __init__(self, keys, raises=False):
        self._keys, self._raises = keys, raises
        self.patterns = []

    def keys(self, pattern):
        self.patterns.append(pattern)
        if self._raises:
            raise ConnectionError("redis down")
        return self._keys


def test_ready_extracts_instance_ids():
    r = FakeRedis([f"{READY_KEY_PREFIX}uuid-a", f"{READY_KEY_PREFIX}uuid-b"])

    assert ready_instance_ids(r) == frozenset({"uuid-a", "uuid-b"})


def test_ready_handles_bytes_keys():
    """redis-py returns bytes unless the client was built with decode_responses.

    Getting this wrong produced a live outage once already -- the log relay shipped
    bytes into send_text and died inside uvicorn (P1.3 finding 7).
    """
    r = FakeRedis([f"{READY_KEY_PREFIX}uuid-a".encode()])

    assert ready_instance_ids(r) == frozenset({"uuid-a"})


def test_ready_scans_only_the_marker_namespace():
    r = FakeRedis([])
    ready_instance_ids(r)

    assert r.patterns == [f"{READY_KEY_PREFIX}*"]


def test_ready_failure_propagates():
    """An empty set past the deadline is a licence to delete the whole fleet.

    So a Redis blip must skip the round rather than be mistaken for "nothing has
    ever registered" -- the same reasoning as serving_count, with more at stake.
    """
    with pytest.raises(ConnectionError):
        ready_instance_ids(FakeRedis([], raises=True))


# --- unclaimed submission ages ---------------------------------------------


def _job(minutes_old, *, aware=False, created=True):
    """A RUNNING submission document with no claim on it."""
    doc = {"_id": f"job-{minutes_old}"}
    if created:
        when = datetime.now(timezone.utc) - timedelta(minutes=minutes_old)
        doc["created"] = when if aware else when.replace(tzinfo=None)
    return doc


def test_unclaimed_ages_measures_from_created():
    db = FakeGirder([_job(5)])
    (age,) = unclaimed_submission_ages(db)
    assert timedelta(minutes=4) < age < timedelta(minutes=6)


def test_unclaimed_ages_handles_naive_and_aware_alike():
    """Girder stores naive UTC; a tz-aware document must not be mis-aged by 5 hours.

    The whole reason this signal returns ages rather than timestamps -- doing the
    subtraction here, where the convention is known, instead of in plan.decide()
    where `now` may be naive local time.
    """
    naive, = unclaimed_submission_ages(FakeGirder([_job(10)]))
    aware, = unclaimed_submission_ages(FakeGirder([_job(10, aware=True)]))
    assert abs(naive - aware) < timedelta(seconds=5)


def test_unclaimed_query_selects_running_and_unclaimed_only():
    """Pins the null-equality match, which is load-bearing rather than incidental.

    Verified against mongo:4.4: equality-to-null matches a missing field, an empty
    ``meta`` and an explicit null, while ``{"$exists": False}`` -- the tempting way to
    write this "properly" -- misses the explicit null and would hide exactly the kind
    of unserved submission this signal exists to surface.
    """
    db = FakeGirder([_job(5)])
    unclaimed_submission_ages(db)
    (_, query, _), *_ = db.calls
    assert query["type"] == SUBMISSION_TYPE
    assert query["status"] == 2
    assert query["meta.worker_queue"] is None
    assert "$exists" not in repr(query["meta.worker_queue"])


def test_unclaimed_skips_undateable_documents():
    """An undateable job would otherwise read as infinitely old and provision forever."""
    assert unclaimed_submission_ages(FakeGirder([_job(5, created=False)])) == ()


def test_unclaimed_degrades_to_depth_only_rather_than_skipping_the_round():
    """A broken query must not block reaps -- gather() is all-or-nothing.

    Returning () can only under-provision, and the next tick retries; raising would
    skip the whole round, and a round that does not happen is a round that does not
    reap. See test_diagnostics.test_a_broken_diagnostics_query_still_reaps, which
    fails if this ever starts raising again.
    """
    assert unclaimed_submission_ages(FakeGirder([], raises=True)) == ()
