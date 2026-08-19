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
    TARGETED_ASSIGNMENT_KEY,
    ready_instance_ids,
    serving_count,
    spent_instance_ids,
    targeted_assignment,
    waiting_submissions,
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


# --- waiting submissions: the demand signal and the assigner's work list ----


def _job(minutes_old, *, aware=False, created=True):
    """A RUNNING submission document with no claim on it."""
    doc = {"_id": f"job-{minutes_old}"}
    if created:
        when = datetime.now(timezone.utc) - timedelta(minutes=minutes_old)
        doc["created"] = when if aware else when.replace(tzinfo=None)
    return doc


def test_a_waiting_submission_is_aged_from_created():
    db = FakeGirder([_job(5)])
    (sub,) = waiting_submissions(db)
    assert timedelta(minutes=4) < sub.age < timedelta(minutes=6)


def test_the_submission_is_identified_not_merely_counted():
    """The id is what makes this the assigner's input and not just a number.

    ``_id`` is stringified here rather than left as an ObjectId: it travels into a log
    line and a decision tuple, and the pure function that consumes it must not have to
    know about pymongo's types.
    """
    (sub,) = waiting_submissions(FakeGirder([_job(5)]))
    assert sub.id == "job-5"
    assert isinstance(sub.id, str)


def test_waiting_submissions_come_back_oldest_first():
    """Matches S7's service order, so the log reads the way the decision runs."""
    db = FakeGirder([_job(1), _job(9), _job(4)])
    assert [s.id for s in waiting_submissions(db)] == ["job-1", "job-9", "job-4"], (
        "the fake returns documents in insertion order, so this pins only that the "
        "signal does not reorder them -- the ordering itself is Mongo's, pinned by "
        "test_the_scan_window_keeps_the_oldest_submissions"
    )


def test_the_scan_window_keeps_the_oldest_submissions():
    """Direction plus limit, and the pair is load-bearing once this places work.

    Newest-first was harmless while the result was only counted. As the assigner's
    work list it would truncate away the head of the line: past CLAIM_SCAN_LIMIT
    waiting submissions, the oldest -- the very one S7 promises to serve next --
    becomes invisible to the controller and starves.
    """
    db = FakeGirder([_job(5)])
    waiting_submissions(db)
    sorts = [c for c in db.calls if c[0] == "sort"]
    assert sorts == [("sort", "created", 1)], "oldest first, exactly once"
    assert ("limit", 100) in db.calls


def test_waiting_ages_handle_naive_and_aware_alike():
    """Girder stores naive UTC; a tz-aware document must not be mis-aged by 5 hours.

    The whole reason this signal returns ages rather than timestamps -- doing the
    subtraction here, where the convention is known, instead of in plan.decide()
    where `now` may be naive local time.
    """
    naive, = waiting_submissions(FakeGirder([_job(10)]))
    aware, = waiting_submissions(FakeGirder([_job(10, aware=True)]))
    assert abs(naive.age - aware.age) < timedelta(seconds=5)


def test_unclaimed_query_selects_running_and_unclaimed_only():
    """Pins the null-equality match, which is load-bearing rather than incidental.

    Verified against mongo:4.4: equality-to-null matches a missing field, an empty
    ``meta`` and an explicit null, while ``{"$exists": False}`` -- the tempting way to
    write this "properly" -- misses the explicit null and would hide exactly the kind
    of unserved submission this signal exists to surface.
    """
    db = FakeGirder([_job(5)])
    waiting_submissions(db)
    (_, query, _), *_ = db.calls
    assert query["type"] == SUBMISSION_TYPE
    assert query["status"] == 2
    assert query["meta.worker_queue"] is None
    assert "$exists" not in repr(query["meta.worker_queue"])


def test_unclaimed_skips_undateable_documents():
    """An undateable job would otherwise read as infinitely old and provision forever."""
    assert waiting_submissions(FakeGirder([_job(5, created=False)])) == ()


def test_unclaimed_degrades_to_depth_only_rather_than_skipping_the_round():
    """A broken query must not block reaps -- gather() is all-or-nothing.

    Returning () can only under-provision, and the next tick retries; raising would
    skip the whole round, and a round that does not happen is a round that does not
    reap. See test_diagnostics.test_a_broken_diagnostics_query_still_reaps, which
    fails if this ever starts raising again.
    """
    assert waiting_submissions(FakeGirder([], raises=True)) == ()


# --- the arm flag, read from Girder's own setting --------------------------


class FakeSettings:
    """``db["setting"]`` with one document lookup, recording the query."""

    def __init__(self, doc=None, raises=False):
        self.doc, self.raises, self.queries = doc, raises, []

    def __getitem__(self, name):
        assert name == "setting", "the flag lives in Girder's settings collection"
        return self

    def find_one(self, query):
        if self.raises:
            raise RuntimeError("mongo down")
        self.queries.append(query)
        return self.doc


def test_the_arm_flag_is_read_from_girders_setting():
    """One value, two processes. ``submit_job`` reads this same document."""
    db = FakeSettings({"key": TARGETED_ASSIGNMENT_KEY, "value": True})

    assert targeted_assignment(db) is True
    assert db.queries == [{"key": TARGETED_ASSIGNMENT_KEY}]


def test_an_unwritten_setting_means_off():
    """Girder's default is off, and the two defaults have to agree.

    A deployment that has never set it must not start assigning, because
    ``submit_job`` on the same deployment is still publishing to the shared queue.
    """
    assert targeted_assignment(FakeSettings(None)) is False
    assert targeted_assignment(FakeSettings({"key": TARGETED_ASSIGNMENT_KEY})) is False


def test_the_arm_flag_refuses_to_guess_when_mongo_is_down():
    """Both defaults are wrong in the deployment where it matters.

    ``False`` while Girder is armed stalls every submission; ``True`` while it is not
    publishes a second chain for one already dispatched. The caller skips the round on
    a Mongo failure, which is the only safe answer.
    """
    with pytest.raises(RuntimeError):
        targeted_assignment(FakeSettings(raises=True))


def test_a_submission_dispatched_the_old_way_is_demand_but_not_ours_to_place():
    """The mixed-mode case, and getting it wrong is two workers on one workspace.

    Rollout step 3 runs both paths at once. A submission ``submit_job`` published to
    the shared queue is RUNNING and unclaimed, so it is real demand and still needs an
    instance -- but publishing a second chain for it is the failure S2 exists to
    remove, arriving through the operator's door rather than the queue's.
    """
    old = _job(5)
    new = _job(3)
    new["meta"] = {"awaiting_assignment": True}

    placed = {s.id: s.assignable for s in waiting_submissions(FakeGirder([old, new]))}

    assert placed == {"job-5": False, "job-3": True}


def test_the_marker_is_projected_or_every_submission_reads_as_ours():
    """The field has to be asked for; a projection that omits it defaults to False.

    Silent in exactly the wrong direction: every submission would look un-placeable
    and the fleet would boot instances it never assigns anything to.
    """
    db = FakeGirder([_job(5)])
    waiting_submissions(db)

    (find,) = [c for c in db.calls if c[0] == "find"]
    assert "meta.awaiting_assignment" in find[2]
