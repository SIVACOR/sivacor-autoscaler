"""The two inputs the scaling decision needs, read from the systems that know.

Kept separate from :mod:`plan` so the arithmetic stays testable without a broker.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from .plan import RunningJob

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
#: 3. **Reads only -- for now, and this is the sentence that expires.** P0.5's warning
#:    about staying on HTTP applies to *writes*: failing a submission has to fire
#:    ``jobs.job.update.after``, which is bound only in the Girder server process.
#:    Nothing here writes, so that constraint does not bind -- but S2/S3 of
#:    ``worker_sizing_plan.md`` make this process the *assigner*, and its first
#:    ``meta.worker_queue`` write ends that. The replacement rule is already known
#:    rather than guessed: it was measured in the deployed controller container on
#:    2026-08-19 (that plan's open item 6).
#:
#:    * Girder's model layer needs nothing bootstrapped beyond ``GIRDER_MONGO_URI``:
#:      ``config.getConfig()`` resolves the database uri from it, with no ``girder.cfg``
#:      and no plugin load.
#:    * ``SIVACORPlugin.load()`` never runs here, so **not one girder_sivacor handler is
#:      bound**. Girder *core* handlers are -- the models bind them themselves -- so "no
#:      event fires in the controller" is the wrong way to say it, and saying it that way
#:      invites someone to trust a model call whose side effect the plugin owns.
#:    * So: a raw ``update_one`` on ``meta.*`` is safe from here, and ``updateJob()`` is
#:      not. Job *status*, the submission folder's status and the researcher's email stay
#:      Girder's alone -- from this process they would silently not happen.
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


def running_jobs_by_instance(db, queue_prefix: str = "sivacor") -> dict[str, RunningJob]:
    """Instance id -> the submission Girder still believes is executing on it.

    Same ``meta.worker_queue`` join as :func:`spent_instance_ids`, narrowed to RUNNING
    jobs. It answers the one question the reap step could not previously ask: *was
    anyone still using this VM?* A SHUTOFF instance with a RUNNING submission on it is
    a worker that died mid-run, not one that finished.

    **Never raises, and an empty result is safe.** Unlike every other signal here this
    one feeds no arithmetic -- it only decides how a reap is worded and whether the
    console log is captured first. A controller that stopped reaping because a
    *diagnostics* query failed would trade a leaked instance for a better error
    message, which is the wrong trade. Callers get ``{}`` and the old behaviour.
    """
    try:
        jobs = list(
            db[JOB_COLLECTION]
            .find(
                {"type": SUBMISSION_TYPE, "status": JOB_RUNNING},
                {"meta.worker_queue": 1, "meta.heartbeat": 1},
            )
            .sort("created", -1)
            .limit(CLAIM_SCAN_LIMIT)
        )
    except Exception:
        logger.warning("Could not read running submissions per instance", exc_info=True)
        return {}

    prefix = f"{queue_prefix}."
    out: dict[str, RunningJob] = {}
    for job in jobs:
        meta = job.get("meta") or {}
        queue = meta.get("worker_queue")
        if not queue or not queue.startswith(prefix):
            continue
        out[queue[len(prefix):]] = RunningJob(
            id=str(job.get("_id")), heartbeat=meta.get("heartbeat")
        )
    logger.debug("running submissions by instance: %s", sorted(out))
    return out


def unclaimed_submission_ages(db) -> tuple[timedelta, ...]:
    """How long each RUNNING submission has gone without any worker claiming it.

    The complement of :func:`spent_instance_ids`: submissions Girder has RUNNING that
    carry no ``meta.worker_queue`` at all. Every one of them is a submission nobody is
    working on, and each needs a worker -- which is demand that
    :func:`queue_depth` **cannot** see.

    Why depth is not enough. Depth counts messages *sitting in the Redis list*. A
    message a worker has already reserved is gone from that list but not yet executing,
    so it is invisible to depth while being just as unserved. Production 2026-08-12: a
    worker restarted mid-chain by the wedge supervisor re-subscribed to the dispatch
    queue (``cancel_consumer`` does not survive a restart), reserved the next
    submission's head task it could not start for hours, and the controller read
    ``depth=0, 0 available of 1 live`` for 17 minutes and created nothing. Same stall
    class as run 3 / run 5, reached through yet another path -- and, like both of those,
    a bad *input* to correct arithmetic rather than bad arithmetic.

    **Returns ages, not timestamps, on purpose.** Girder stores ``created`` as naive
    UTC while the controller's ``now`` follows OpenStack and is tz-aware -- except when
    the fleet is empty, which is exactly the scale-from-zero case that matters most.
    Subtracting those two raises, and coercing either one means guessing a zone the
    value does not carry (the autoscaler's own container logs in America/Chicago, so
    the guess would be five hours wrong). Doing the subtraction here, where Girder's
    convention is known, removes the guess entirely and leaves :func:`plan.decide`
    comparing timedeltas.

    **Swallows its own errors and returns ``()``**, unlike :func:`serving_count`, which
    raises. The first instinct was to raise here too -- an empty reading is, after all,
    exactly the stall this exists to correct -- but that is the wrong trade and the test
    suite says so: ``gather()`` is all-or-nothing, so a raise skips the whole round, and
    *a round that does not happen is a round that does not reap*. Blocking reaps on a
    Mongo blip would trade a leaked instance and a destroyed console buffer for a
    slightly faster scale-up.

    The asymmetry that makes this safe: returning ``()`` degrades to the depth-only
    reading -- the behaviour before this signal existed -- and can only ever
    *under*-provision. ``serving_count`` returning 0 would invent headroom and could
    provoke a burst of instances, which is why that one refuses to guess. Failing loudly
    in the log is what covers a persistent outage.
    """
    now = datetime.now(timezone.utc)
    try:
        # ``{"meta.worker_queue": None}`` deliberately relies on Mongo's equality-to-null
        # matching a *missing* field as well as an explicit one. "Unclaimed" means no
        # queue recorded, and all three shapes mean that. Measured on mongo:4.4::
        #
        #     {"meta.worker_queue": None}            -> no meta, meta {}, explicit null
        #     {"meta.worker_queue": {"$exists": 0}}  -> no meta, meta {}
        #     {"$eq": None, "$exists": True}         -> explicit null
        #
        # So ``$exists: False`` -- the obvious way to be explicit -- is strictly
        # *narrower*: it drops a submission whose field was nulled rather than never
        # written, which would put it back in the invisible-demand class this signal
        # exists to end. In practice claim() only ever ``$set``s a string, so today the
        # field is simply absent; the wider match costs nothing and survives a future
        # code path that clears it.
        jobs = list(
            db[JOB_COLLECTION]
            .find(
                {
                    "type": SUBMISSION_TYPE,
                    "status": JOB_RUNNING,
                    "meta.worker_queue": None,
                },
                {"created": 1},
            )
            .sort("created", -1)
            .limit(CLAIM_SCAN_LIMIT)
        )
    except Exception:
        logger.warning(
            "Could not read unclaimed submissions; scaling on queue depth alone this "
            "round, which cannot see a submission whose message a worker has reserved",
            exc_info=True,
        )
        return ()

    ages: list[timedelta] = []
    for job in jobs:
        created = job.get("created")
        if created is None:
            # Nothing to age it against; counting it would make an undateable
            # document look infinitely old and provision on every tick forever.
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        ages.append(now - created)
    if ages:
        logger.debug("unclaimed submission ages: %s", sorted(ages))
    return tuple(ages)
