"""Executing an assignment: claim, then publish.

The ordering is the whole subject. Claim-then-publish can leave a submission bound to
a worker that was never told, which the server-side reaper cleans up. The reverse, or
a rollback after a failed publish, can put **two workers on one workspace** -- the
failure targeted assignment exists to remove. One of those is recoverable and the
other is not, so every test here is about staying on the recoverable side.

Girder is imported lazily by :mod:`dispatch`, so these run in a bare virtualenv with
no Girder installed: the publish path is exercised by injecting a fake builder into
the module the lazy import resolves to.
"""

import sys
import types
from datetime import datetime, timezone

import pytest
from bson import ObjectId

from sivacor_autoscaler import dispatch

SUB = str(ObjectId())


class FakeJobs:
    """``db["job"]`` with a one-shot atomic claim, like Mongo's."""

    def __init__(self, claimable=True, awaiting=True):
        self.claimed = None
        self.queries = []
        self._doc = {
            "_id": ObjectId(SUB),
            "type": "sivacor_submission",
            "userId": "u1",
            "meta": {"awaiting_assignment": awaiting},
            "sivacorChain": {"fileId": "f1", "stages": [{"main_file": "m.R"}],
                             "secrets": {"encrypted_secrets": "x"}},
        }
        self._claimable = claimable

    def find_one_and_update(self, query, update, return_document=None):
        self.queries.append(query)
        if not self._claimable or self.claimed is not None:
            return None
        if query.get("meta.awaiting_assignment") and not self._doc["meta"].get(
            "awaiting_assignment"
        ):
            return None
        self.claimed = update["$set"]
        doc = dict(self._doc)
        doc["meta"] = {**self._doc["meta"], **update["$set"]}
        return doc


class FakeDb:
    def __init__(self, jobs):
        self.jobs = jobs

    def __getitem__(self, name):
        assert name == "job"
        return self.jobs


# --- the queue name --------------------------------------------------------


def test_the_queue_name_is_the_one_the_worker_derives_for_itself():
    """Four places join on this string shape; inventing a second identity breaks all.

    ``worker-cloud-init.sh`` builds it from the metadata service, and both
    ``spent_instance_ids`` and ``running_jobs_by_instance`` recover the instance id by
    stripping the prefix back off.
    """
    assert dispatch.queue_for("uuid-a", "sivacor") == "sivacor.uuid-a"
    assert dispatch.queue_for("uuid-a", "renamed") == "renamed.uuid-a"


# --- the claim -------------------------------------------------------------


def test_the_claim_is_atomic_and_wins_exactly_once():
    """A controller that dies mid-tick and re-decides must not bind twice.

    The predicate carries the unassigned condition, so the second call matches nothing.
    Proven against the mirror's real database on 2026-08-19 (open item 6).
    """
    jobs = FakeJobs()

    assert dispatch.claim(FakeDb(jobs), SUB, "sivacor.a") is not None
    assert dispatch.claim(FakeDb(jobs), SUB, "sivacor.a") is None

    (first, _) = jobs.queries
    assert first["meta.worker_queue"] is None, "the claim must select on being unclaimed"


def test_the_claim_refuses_a_submission_girder_already_dispatched():
    """Belt to WaitingSubmission.assignable's braces, and the belt is the atomic one.

    During rollout both paths are live. Publishing a chain for a submission
    ``submit_job`` already sent to the shared queue is two workers on one workspace,
    so the check that matters sits inside the update, not in the arithmetic that
    proposed it.
    """
    jobs = FakeJobs(awaiting=False)

    assert dispatch.claim(FakeDb(jobs), SUB, "sivacor.a") is None


def test_the_claim_writes_assigned_at_in_the_same_update():
    """Two updates would leave a window; and without it the reaper fails the run.

    A submission that waited an hour has an hour-old ``updated`` the moment it is
    handed a worker, so ``meta.assigned_at`` is both its sign of life and the start of
    its runtime clock. The server-side reaper reads exactly this field.
    """
    jobs = FakeJobs()
    now = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)

    dispatch.claim(FakeDb(jobs), SUB, "sivacor.a", now=now)

    assert jobs.claimed == {"meta.worker_queue": "sivacor.a", "meta.assigned_at": now}


# --- publishing ------------------------------------------------------------


#: Stands in for a File document; identity is all the fake builder needs.
A_FILE = {"_id": "f1", "name": "package.zip"}


def _fake_girder(monkeypatch, *, api_url="https://girder.example/api/v1", fail=False,
                 file=A_FILE, sent=None):
    """Install the minimum of Girder that dispatch.publish imports lazily."""
    mods = {}

    def module(name, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        mods[name] = mod
        return mod

    class Setting:
        def get(self, key):
            return api_url

    class File:
        def load(self, _id, force=False):
            return file

    class Head:
        """The chain's first signature: options, args and a recording ``set``."""

        def __init__(self):
            self.options = {"girder_job_title": "Moving package.zip"}
            self.args = ("u1", "f1", [], "j1", 60)
            self.kwargs = {}

        def set(self, **kw):
            if sent is not None:
                sent.append(("head.set", kw))

        def freeze(self):
            # celery assigns an id only if the signature has none, so the id a
            # frozen signature reports is the id the message will carry.
            return types.SimpleNamespace(id="celery-task-id-1")

    class Chain:
        def __init__(self):
            self.tasks = [Head()]

        def apply_async(self, queue=None):
            if fail:
                raise RuntimeError("broker unreachable")
            if sent is not None:
                sent.append(queue)

    def build_submission_chain(job, file_, stages, secrets):
        if sent is not None:
            sent.append((job["_id"], file_, stages, secrets))
        return Chain()

    class Job:
        def createJob(self, **kw):
            if sent is not None:
                sent.append(
                    ("child job", kw["type"], kw["title"], kw["args"], kw.get("otherFields"))
                )
            return {"_id": "child-1"}

    class Users:
        def load(self, _id, force=False):
            return {"_id": _id}

    module("girder")
    module("girder.models")
    module("girder.models.file", File=File)
    module("girder.models.user", User=Users)
    module("girder_jobs")
    module("girder_jobs.models")
    module("girder_jobs.models.job", Job=Job)
    module("girder.models.setting", Setting=Setting)
    module("girder_plugin_worker")
    module("girder_plugin_worker.constants",
           PluginSettings=types.SimpleNamespace(API_URL="worker.api_url"))
    module("girder_plugin_worker.utils", jobInfoSpec=lambda job: {"jobId": job["_id"]})
    module("girder_sivacor")
    module("girder_sivacor.rest", build_submission_chain=build_submission_chain)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_publishing_uses_girders_own_builder_and_the_stashed_inputs(monkeypatch):
    """One builder, imported rather than reimplemented.

    A second copy of the chain in this repo is a scheduling policy that can drift from
    the one ``submit_job`` uses -- and it would drift silently, because both produce a
    chain that runs.
    """
    sent = []
    _fake_girder(monkeypatch, sent=sent)
    jobs = FakeJobs()
    job = dispatch.claim(FakeDb(jobs), SUB, "sivacor.a")

    dispatch.publish(job, "sivacor.a")

    built = sent[0]
    published = sent[-1]
    assert built[2] == [{"main_file": "m.R"}], "the stages Girder validated, unchanged"
    assert built[3] == {"encrypted_secrets": "x"}, "already encrypted; not re-derived"
    assert published == "sivacor.a", "to that instance's queue and nowhere else"


def test_publishing_refuses_when_the_callback_url_is_unset(monkeypatch):
    """Silent and late otherwise: every step would be published with no way back.

    ``getWorkerApiUrl`` falls back to cherrypy's request context, which does not exist
    in this process, so an unset setting yields a chain whose steps cannot POST their
    child jobs, log or progress. Nothing would look wrong until a researcher's run
    produced no output.
    """
    _fake_girder(monkeypatch, api_url="")
    job = dispatch.claim(FakeDb(FakeJobs()), SUB, "sivacor.a")

    with pytest.raises(RuntimeError, match="worker.api_url"):
        dispatch.publish(job, "sivacor.a")


def test_publishing_refuses_a_job_with_nothing_stashed(monkeypatch):
    """The two flags disagree: Girder built this one for the shared queue."""
    _fake_girder(monkeypatch)

    with pytest.raises(RuntimeError, match="sivacorChain"):
        dispatch.publish({"_id": "j", "meta": {}}, "sivacor.a")


def test_a_vanished_file_is_an_error_not_an_empty_run(monkeypatch):
    """The researcher deleted the upload between submitting and being placed."""
    _fake_girder(monkeypatch, file=None)
    job = dispatch.claim(FakeDb(FakeJobs()), SUB, "sivacor.a")

    with pytest.raises(RuntimeError, match="gone"):
        dispatch.publish(job, "sivacor.a")


# --- the two halves together -----------------------------------------------


def test_assign_claims_before_it_publishes(monkeypatch):
    """The ordering, asserted directly rather than inferred from the call sites.

    Publishing first can send the same chain twice if the tick dies between the two
    steps, and the duplicate is undetectable: both messages are valid.
    """
    order = []
    jobs = FakeJobs()
    monkeypatch.setattr(
        dispatch, "claim", lambda *a, **k: order.append("claim") or {"_id": "j"}
    )
    monkeypatch.setattr(dispatch, "publish", lambda *a: order.append("publish"))

    assert dispatch.assign(FakeDb(jobs), SUB, "uuid-a", "sivacor") is True
    assert order == ["claim", "publish"]


def test_a_failed_publish_does_not_release_the_claim(monkeypatch):
    """The tempting rollback, and it is the unrecoverable direction.

    ``apply_async`` can fail with the message already on the broker, so releasing the
    claim offers the submission for a second placement: two workers, one workspace. A
    submission stuck claimed is failed by the server-side reaper instead -- wrong in
    its wording, right in its effect, and recoverable.
    """
    _fake_girder(monkeypatch, fail=True)
    jobs = FakeJobs()

    assert dispatch.assign(FakeDb(jobs), SUB, "uuid-a", "sivacor") is False
    assert jobs.claimed == {
        "meta.worker_queue": "sivacor.uuid-a",
        "meta.assigned_at": jobs.claimed["meta.assigned_at"],
    }, "the claim stands: never offer a possibly-published submission twice"


def test_losing_the_claim_is_not_an_error(monkeypatch):
    """Something else got there first, or it belongs to the other dispatch path.

    The instance simply stays free and is offered again next tick, so this must not
    raise: raising would skip the remaining bindings in the same round.
    """
    _fake_girder(monkeypatch)
    jobs = FakeJobs(claimable=False)

    assert dispatch.assign(FakeDb(jobs), SUB, "uuid-a", "sivacor") is False


def test_the_head_step_gets_a_child_job_this_process_creates(monkeypatch):
    """Open item 7, answered: girder_worker will NOT create it when we publish.

    ``get_context()`` branches on ``cherrypy.request.app``. Outside a REST request it
    takes the non-Girder branch, which needs a *running* celery task to POST the child
    job -- a worker publishing the next link has one, this process does not. It raises
    ``MissingJobArguments('Parent task is None')``, logs it and carries on, so the
    submission runs and only the child job silently fails to exist. That document is
    how ``GET /job/:id/children`` finds the head (by ``args.3``), so without this the
    researcher's monitor loses its first step.

    ``jobInfoSpec`` on the head signature is what makes the handler skip its own
    attempt rather than warn about it.
    """
    sent = []
    _fake_girder(monkeypatch, sent=sent)
    job = dispatch.claim(FakeDb(FakeJobs()), SUB, "sivacor.a")

    dispatch.publish(job, "sivacor.a")

    child = next(e for e in sent if isinstance(e, tuple) and e[0] == "child job")
    assert child[1] == "celery", "get_submission_child_jobs selects on type"
    assert child[3][3] == "j1", "args.3 is the join back to the submission"
    # Not optional, and omitting it is worse than creating no child job at all:
    # girder_plugin_worker's attachParentJob does
    #     event.info['parentId'] = Job().findOne({'celeryTaskId': ...})['_id']
    # with no null check, so every LATER step's POST /job 500s and the submission
    # ends up with one child job instead of thirteen. Mirror, 2026-08-19.
    assert child[4] == {"celeryTaskId": "celery-task-id-1"}

    spec = next(e for e in sent if isinstance(e, tuple) and e[0] == "head.set")
    # In *headers*, not bare options: jobInfoSpec is neither a reserved header nor a
    # reserved option, so Task.apply_async would never copy it into the message, and
    # both the publish handler and the running task read it from there.
    assert "jobInfoSpec" in spec[1]["headers"]
