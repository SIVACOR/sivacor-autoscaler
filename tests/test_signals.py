"""Tests for the Girder-side signals.

Two kinds of case here. Most parse a marker written by another repo, so they are
about being unsurprised by what Girder actually returns. The rest pin the *endpoint*,
which is not decoration: reading the wrong one returns ``200`` and an empty list, so
both signals fail silently as "nothing running, nothing spent". A stub cannot tell
the two endpoints apart -- these assert the request, which is the one thing about it
a stub can still verify. See :data:`~sivacor_autoscaler.signals.JOB_LIST_ENDPOINT`.
"""

import pytest

from sivacor_autoscaler.signals import (
    JOB_LIST_ENDPOINT,
    serving_count,
    spent_instance_ids,
)


class FakeGirder:
    def __init__(self, jobs):
        self.jobs = jobs
        self.calls = []

    def get(self, path, parameters=None):
        self.calls.append((path, parameters))
        return self.jobs


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


def test_query_is_bounded_and_scoped():
    """An unbounded job scan would grow without limit over a pilot's lifetime."""
    client = FakeGirder([])
    spent_instance_ids(client)
    _, params = client.calls[0]

    assert params["types"] == '["sivacor_submission"]'
    assert params["limit"] > 0


def test_girder_failure_propagates():
    """Returning an empty set would read as 'every worker is available'."""

    class Broken:
        def get(self, *a, **kw):
            raise RuntimeError("girder down")

    with pytest.raises(RuntimeError):
        spent_instance_ids(Broken())


# -- the endpoint ---------------------------------------------------------------
#
# These look trivial and are not. Both signals asked `GET /job` until 2026-08-02,
# which defaults `userId` to the authenticated user and so returned `[]` -- every
# worker looked available and every submission looked idle, with a 200 and no error
# anywhere. The controller could then only create capacity when the fleet was empty,
# and one submission waited 18 min 09 s. An earlier revision of the first test below
# asserted `path == "job"`, pinning the bug rather than catching it.


def test_spent_reads_the_all_users_endpoint():
    """Submissions are owned by researchers, not by the admin the controller uses."""
    client = FakeGirder([])
    spent_instance_ids(client)
    path, _ = client.calls[0]

    assert path == JOB_LIST_ENDPOINT
    assert path == "job/all", "GET /job silently scopes to the authenticated user"


def test_serving_reads_the_all_users_endpoint():
    """The same trap, in the signal that was wrong for the controller's whole life."""
    client = FakeGirder([])
    serving_count(client)
    path, _ = client.calls[0]

    assert path == JOB_LIST_ENDPOINT
    assert path == "job/all", "GET /job silently scopes to the authenticated user"


# -- serving_count --------------------------------------------------------------


def test_serving_counts_running_submissions():
    client = FakeGirder([job(), job(), job()])

    assert serving_count(client) == 3


def test_serving_filters_to_running_submissions():
    """Terminal submissions are not capacity, so the query must do the filtering."""
    client = FakeGirder([])
    serving_count(client)
    _, params = client.calls[0]

    assert params["types"] == '["sivacor_submission"]'
    assert params["statuses"] == "[2]"


def test_serving_failure_propagates():
    """Returning 0 would read as 'nothing is running' and could provoke a burst."""

    class Broken:
        def get(self, *a, **kw):
            raise RuntimeError("girder down")

    with pytest.raises(RuntimeError):
        serving_count(Broken())
