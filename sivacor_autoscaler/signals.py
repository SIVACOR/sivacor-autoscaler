"""The two inputs the scaling decision needs, read from the systems that know.

Kept separate from :mod:`plan` so the arithmetic stays testable without a broker.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Girder's numeric code for a RUNNING job (girder_jobs.constants.JobStatus.RUNNING).
JOB_RUNNING = 2

#: Girder's collection of job documents, and the field the plugin marks claims in.
#: Read straight from MongoDB rather than over the REST API, for three reasons:
#:
#: 1. **No credential to bootstrap.** The REST path needed an admin ``GIRDER_API_KEY``,
#:    which cannot exist until after Girder is first started -- so a fresh deployment
#:    could not bring the stack up in one pass. The controller is pinned to the manager
#:    (it bind-mounts the worker template and clouds.yaml), so it is a peer of Mongo
#:    exactly as ``local_worker`` is; the same reasoning retired ``GIRDER_API_KEY`` from
#:    the housekeeping sweeps in P0.5.
#: 2. **No endpoint semantics to get wrong.** ``GET /job`` defaults its ``userId`` to the
#:    authenticated caller, and submissions belong to the researchers who made them --
#:    so it returned ``200`` and ``[]`` on every tick, silently, for the controller's
#:    entire life. That cost an 18 min 09 s stall before anyone noticed (plan run 5).
#:    A query has no such hidden scoping.
#: 3. **Reads only.** P0.5's warning about staying on HTTP applies to *writes*: failing a
#:    submission has to fire ``jobs.job.update.after``, which is bound only in the Girder
#:    server process. Nothing here writes, so that constraint does not bind.
#:
#: The cost is coupling to Girder's schema instead of its API. Mild: the collection and
#: the ``type``/``status`` fields are stable, and ``meta.worker_queue`` is SIVACOR's own.
JOB_COLLECTION = "job"

#: Girder's job type for a submission, set by ``rest.py``'s ``submit_job``.
SUBMISSION_TYPE = "sivacor_submission"


def queue_depth(redis_client, queue: str) -> int:
    """Submissions published to ``queue`` that no worker has taken yet.

    Celery's Redis transport stores each queue as a list under its own name, so
    ``LLEN`` is the count of unclaimed messages. A message being *executed* is not in
    the list, which is exactly the semantics wanted here: this is "waiting for a
    worker", not "in the system".
    """
    depth = int(redis_client.llen(queue) or 0)
    logger.debug("queue %s depth=%d", queue, depth)
    return depth


def serving_count(db) -> int:
    """Submissions currently executing.

    Load-bearing, not decorative: an ephemeral worker stops consuming the dispatch
    queue as soon as it accepts a submission, so busy instances will never absorb the
    queue and must be counted separately or the fleet deadlocks. See
    :func:`plan.decide`.

    Read from Girder's database rather than by broadcasting ``celery inspect``: a
    broadcast is slow, needs every worker to answer, and silently under-reports when
    one is wedged -- which is precisely when the number matters. See
    :data:`JOB_COLLECTION` for why it is the database and not the REST API.
    """
    try:
        return db[JOB_COLLECTION].count_documents(
            {"type": SUBMISSION_TYPE, "status": JOB_RUNNING}
        )
    except Exception:
        # Returning 0 here would look like "nothing is running" and could provoke a
        # burst of instances, so refuse to guess and let the caller skip the round.
        logger.warning("Could not read running submissions from Girder", exc_info=True)
        raise


#: How many recent submissions to scan for worker claims. Only *live* instances
#: matter, there are at most ``max_instances`` of them, and each takes one or two
#: submissions, so a few dozen is already generous. Bounded because this runs every
#: tick and an unbounded job scan would grow without limit over a pilot's lifetime.
CLAIM_SCAN_LIMIT = 100


#: Redis key prefix the worker plugin writes readiness markers under. Must match
#: ``girder_sivacor.worker_plugin.run_submission.READY_KEY_PREFIX``.
READY_KEY_PREFIX = "sivacor:ready:"


def ready_instance_ids(redis_client) -> frozenset[str]:
    """Instances whose celery worker started and reached the broker (plan D9).

    An instance that boots but fails to provision is neither *spent* nor *serving*,
    so without this it reads as available capacity forever -- and it cannot reclaim
    itself either, because the self-shutdown supervisor is written by the same
    script that failed. Measured 2026-08-02: one lost dpkg-lock race stranded a VM
    for a whole run and stalled a submission behind it.

    The marker is written once, from the worker's ``worker_ready`` handler, so its
    presence means celery actually started *and* connected -- strictly stronger than
    "the provisioning script finished", which a wrong docker GID or a bad broker
    password would satisfy with a dead worker.

    **Raises rather than returning an empty set on failure.** Empty means "no
    instance has ever registered", which past the boot deadline is a licence to
    delete the entire fleet -- so a Redis blip must skip the round, not act on a
    guess. Same reasoning as :func:`serving_count`, with more at stake.
    """
    try:
        keys = redis_client.keys(f"{READY_KEY_PREFIX}*")
    except Exception:
        logger.warning("Could not read readiness markers from Redis", exc_info=True)
        raise
    ready = {
        (k.decode() if isinstance(k, bytes) else k)[len(READY_KEY_PREFIX):]
        for k in keys
    }
    logger.debug("ready instances (celery registered): %s", sorted(ready))
    return frozenset(ready)


def spent_instance_ids(db, queue_prefix: str = "sivacor") -> frozenset[str]:
    """Instance ids that have already claimed a submission, and so are spent.

    An ephemeral worker stops consuming the dispatch queue the instant it accepts a
    submission, and powers off when finished. Counting such an instance as capacity
    is what made the controller refuse to create the instance a queued submission
    needed -- a 4 min 45 s stall observed 2026-08-01, and ~13 min had a later
    submission not happened to bump the queue depth. See :func:`plan.decide`.

    The mapping is direct because a worker's private queue is named
    ``sivacor.<instance-uuid>`` (``worker-cloud-init.sh``), so the marker
    ``prepare_submission`` writes to ``meta.worker_queue`` names the instance.

    Read from Girder's database rather than ``celery inspect active_queues``, which
    would answer the same question over the broker. A worker whose broker connection
    has died cannot answer a broadcast -- and that is exactly the situation where this
    number decides whether a submission gets an instance. The marker is written once,
    by the worker, at claim time; nothing has to be reachable afterwards for it to
    stay true. See :data:`JOB_COLLECTION` for why it is the database, not the API.

    Newest first and bounded: only *live* instances matter and there are at most
    ``max_instances`` of them, so the scan window only has to be deep enough to cover
    the submissions they are working on.
    """
    jobs = (
        db[JOB_COLLECTION]
        .find({"type": SUBMISSION_TYPE}, {"meta.worker_queue": 1})
        .sort("created", -1)
        .limit(CLAIM_SCAN_LIMIT)
    )
    prefix = f"{queue_prefix}."
    spent = {
        queue[len(prefix):]
        for job in jobs
        if (queue := (job.get("meta") or {}).get("worker_queue"))
        and queue.startswith(prefix)
    }
    logger.debug("spent instances (claimed a submission): %s", sorted(spent))
    return frozenset(spent)
