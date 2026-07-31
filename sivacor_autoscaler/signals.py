"""The two inputs the scaling decision needs, read from the systems that know.

Kept separate from :mod:`plan` so the arithmetic stays testable without a broker.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Girder's numeric code for a RUNNING job (girder_jobs.constants.JobStatus.RUNNING).
JOB_RUNNING = 2


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


def serving_count(girder_client) -> int:
    """Submissions currently executing.

    Load-bearing, not decorative: an ephemeral worker stops consuming the dispatch
    queue as soon as it accepts a submission, so busy instances will never absorb the
    queue and must be counted separately or the fleet deadlocks. See
    :func:`plan.decide`.

    Read from Girder rather than by broadcasting ``celery inspect``: a broadcast is
    slow, needs every worker to answer, and silently under-reports when one is
    wedged -- which is precisely when the number matters.
    """
    try:
        jobs = girder_client.get(
            "job",
            parameters={
                "types": '["sivacor_submission"]',
                "statuses": f"[{JOB_RUNNING}]",
                "limit": 0,
            },
        )
        return len(jobs)
    except Exception:
        # Returning 0 here would look like "nothing is running" and could provoke a
        # burst of instances, so refuse to guess and let the caller skip the round.
        logger.warning("Could not read running submissions from Girder", exc_info=True)
        raise
