"""Executing an assignment: claim the submission, then publish its chain.

This is the write half of S3 in ``worker_sizing_plan.md``, and the only place this
process writes to Girder at all. Everything it does is deliberately small:

1. one atomic ``find_one_and_update`` on ``meta.*`` -- byte for byte what
   ``rest.py``'s ``claim`` endpoint already does, and for the same reason it
   bypasses ``updateJob()``;
2. one ``File().load``;
3. ``girder_sivacor.rest.build_submission_chain`` -- the *same* builder
   ``submit_job`` uses, imported rather than reimplemented -- published to one queue.

**The rule this module exists to keep, and it is not obvious from the code.**
``SIVACORPlugin.load()`` never runs here, so not one ``girder_sivacor`` event handler
is bound. Girder *core* handlers are, bound by the models themselves -- so "no events
fire in the controller" is the wrong way to say it, and saying it that way invites
someone to trust a model call whose side effect the plugin owns. The accurate rule:
**a raw update on ``meta.*`` is safe from here; ``updateJob()`` is not.** Job status,
the submission folder's status and the researcher's email stay Girder's alone -- from
this process they would silently not happen. Verified in the deployed container on
2026-08-19 (that plan's open item 6).

Girder is imported lazily, inside the functions that need it. The arithmetic in
:mod:`plan` and the signals in :mod:`signals` must stay testable in a bare virtualenv;
only this file needs the girder-sivacor image P0.5 consolidated onto.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from bson import ObjectId
from pymongo import ReturnDocument

from .signals import JOB_COLLECTION, SUBMISSION_TYPE

logger = logging.getLogger(__name__)


def queue_for(instance_id: str, dispatch_queue: str = "sivacor") -> str:
    """The private queue a worker consumes, derived exactly as the worker derives it.

    ``worker-cloud-init.sh`` builds ``sivacor.<instance-uuid>`` from the metadata
    service, and ``routing.QUEUE_PREFIX`` is ``f"{DISPATCH_QUEUE}."``; both
    ``signals.spent_instance_ids`` and ``signals.running_jobs_by_instance`` recover the
    instance id by stripping that prefix back off. One string shape, four places, and
    ``WORKER_QUEUE_OVERRIDE`` stays unused because the controller cannot know the uuid
    before ``create_server`` returns.
    """
    return f"{dispatch_queue}.{instance_id}"


def claim(db, submission_id: str, queue: str, now: datetime | None = None):
    """Bind one submission to one instance's queue. Returns the job document, or None.

    **Atomic, and that is the whole point.** The predicate requires the submission to
    be still unassigned, so a controller that dies mid-tick and re-decides from the
    same state cannot bind it twice: the second call matches nothing and returns
    ``None``. Proven against the mirror's real database on 2026-08-19.

    It also requires ``meta.awaiting_assignment``, which is belt to
    :attr:`plan.WaitingSubmission.assignable`'s braces. That flag is what stops a
    submission ``submit_job`` already published to the shared queue from being
    published a second time here -- two workers, one workspace -- and the check that
    matters is the one inside the atomic update, not the one in the arithmetic.

    ``meta.assigned_at`` is written in the *same* update, never a second one. The
    server-side reaper reads it as both a sign of life and the start of the runtime
    clock: without it a submission that waited an hour has an hour-old ``updated`` the
    moment it is handed a worker, and is failed as abandoned before its first step
    logs.
    """
    return db[JOB_COLLECTION].find_one_and_update(
        {
            "_id": ObjectId(submission_id),
            "type": SUBMISSION_TYPE,
            "meta.worker_queue": None,
            "meta.awaiting_assignment": True,
        },
        {
            "$set": {
                "meta.worker_queue": queue,
                "meta.assigned_at": now or datetime.now(timezone.utc),
            }
        },
        return_document=ReturnDocument.AFTER,
    )


def publish(job, queue: str) -> None:
    """Build this submission's chain and publish it to ``queue``.

    The inputs come from the ``sivacorChain`` stash ``submit_job`` wrote: the file id,
    the validated stages, and the secrets *already encrypted* by the server. Nothing is
    re-validated here and nothing is re-encrypted -- this process holds no user request
    to validate against, and re-deriving either would be a second implementation of a
    decision Girder already made.
    """
    from girder.models.file import File
    from girder.models.setting import Setting
    from girder_plugin_worker.constants import PluginSettings as WorkerSettings
    from girder_sivacor.rest import build_submission_chain

    stash = job.get("sivacorChain")
    if not stash:
        raise RuntimeError(
            "no sivacorChain on the job: it was not created for assignment, so there "
            "is nothing to build. Girder's arm flag and this controller's disagree."
        )
    # Checked here rather than trusted, because the failure is silent and late. Every
    # step's girder_api_url comes from this setting; outside a request there is no
    # cherrypy fallback worth having, so an empty value publishes a chain whose steps
    # cannot call back -- no child jobs, no progress, no log.
    if not Setting().get(WorkerSettings.API_URL):
        raise RuntimeError(
            "worker.api_url is unset; every step of the chain would be published with "
            "no callback URL. Seed it as setup_girder.py does."
        )

    file = File().load(stash["fileId"], force=True)
    if file is None:
        raise RuntimeError(f"file {stash['fileId']} is gone; nothing to run")

    chain = build_submission_chain(job, file, stash["stages"], stash["secrets"])
    _record_head_job(chain, job)
    chain.apply_async(queue=queue)


def _record_head_job(chain, submission) -> None:
    """Create the child job for the chain's first step, which nothing else will.

    **Measured, not assumed, on 2026-08-19** -- this is ``worker_sizing_plan.md``'s
    open item 7, and the answer is that ``girder_before_task_publish`` does *not*
    behave the same when the publisher is not the Girder server:

    ``girder_worker.context.get_context()`` branches on ``cherrypy.request.app``. In a
    REST request it takes the Girder branch and creates the child job through the model
    layer; anywhere else it takes the non-Girder branch, which needs
    ``current_app.current_task`` -- a *running* celery task -- to POST the child job
    over REST. A worker publishing the next link of a chain has one. This process does
    not, so the handler raises ``MissingJobArguments('Parent task is None')``, **logs
    it, and carries on**: the message is published, the submission runs, and only the
    child job silently fails to exist.

    Only the head step is affected -- every later link is published by the worker that
    just ran the previous one -- and the cost is exactly one document, but it is a
    visible one: ``GET /job/:id/children`` finds the head by ``args.3``
    (``rest.py``'s ``get_submission_child_jobs``), so without it the researcher's
    monitor loses its first step, the one they are watching hardest.

    So it is created here, through the model layer this process genuinely has, and
    ``jobInfoSpec`` is attached to the head signature -- which also makes
    ``girder_before_task_publish`` skip its own attempt entirely rather than warn.
    ``celeryTaskId`` is deliberately not set: the id does not exist until publish, and
    nothing in SIVACOR reads that field.
    """
    from girder.models.user import User
    from girder_jobs.models.job import Job
    from girder_plugin_worker.utils import jobInfoSpec

    head = chain.tasks[0]
    child = Job().createJob(
        title=head.options.get("girder_job_title", "Prepare submission"),
        type="celery",
        public=False,
        user=User().load(submission["userId"], force=True),
        args=head.args,
        kwargs=head.kwargs,
    )
    # Into ``headers``, not the signature's options. ``jobInfoSpec`` is neither a
    # reserved header nor a reserved option, so ``Task.apply_async`` would leave a
    # bare ``jobInfoSpec=`` in options and never copy it into the message headers --
    # where both the publish handler and the running task look for it. Setting it the
    # wrong way costs nothing visible: the child job is still created here, the
    # handler still warns, and only the head step's job manager is missing.
    head.set(headers={"jobInfoSpec": jobInfoSpec(child), **head.options.get("headers", {})})


def assign(db, submission_id: str, instance_id: str, dispatch_queue: str) -> bool:
    """Claim then publish, in that order. Returns whether the chain went out.

    **Never the reverse order, and never a rollback.** Claiming second would publish
    the same chain twice if the tick died between the two. Un-claiming after a failed
    publish is the same hazard wearing a helpful face: ``apply_async`` can fail with
    the message already on the broker, so releasing the claim would offer the
    submission for a second placement and put two workers on one workspace. A
    submission stuck claimed is failed by the server-side reaper with a message about
    a lost worker -- wrong in its wording, right in its effect, and recoverable. Two
    workers on one workspace is neither.
    """
    queue = queue_for(instance_id, dispatch_queue)
    job = claim(db, submission_id, queue)
    if job is None:
        # Not an error: something else got there first, or the submission was
        # cancelled, or it belongs to the other dispatch path. The instance stays
        # unassigned and is offered again next tick.
        logger.info(
            "submission %s was already claimed or is not ours to place; %s stays free",
            submission_id,
            instance_id,
        )
        return False
    try:
        publish(job, queue)
    except Exception:
        # Loud, and deliberately without a rollback -- see above. The instance is now
        # spent as far as every signal is concerned, which is correct: its queue holds
        # a submission's name whether or not the chain reached it.
        logger.exception(
            "submission %s is claimed for %s but its chain could not be published; it "
            "will sit until the server-side reaper fails it",
            submission_id,
            queue,
        )
        return False
    logger.info("published submission %s to %s", submission_id, queue)
    return True
