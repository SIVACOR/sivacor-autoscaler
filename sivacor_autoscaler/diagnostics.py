"""Preserve what a worker VM knows, in the last moment before it is deleted.

A fleet worker is a black box by construction: no floating IP, so no ssh; its journal
lives on a disk that is about to be destroyed; and ``openstack console log show`` stops
working the instant ``delete_server`` returns. The controller is therefore the only
component positioned to take a post-mortem, and it has exactly one chance.

This module is deliberately incapable of failing a reap. Every read is best-effort and
every error is swallowed: a leaked instance costs real allocation, an unwritten dump
costs one investigation, and the reap has to win. Nothing here is on the scaling path.

**Treat the output as secret-bearing.** It goes to a file, never to the service log,
because the service log is shared and a console buffer is not curated -- see
:data:`fleet.REDACTED_SERVER_FIELDS` for the specific hazard and the incident behind it.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from . import fleet

logger = logging.getLogger(__name__)

#: Dumps are written 0600 and their directory 0700: a console buffer is uncurated
#: output from a machine that was handling a researcher's data.
_FILE_MODE = 0o600
_DIR_MODE = 0o700


def capture(conn, directory: Path, instance, why: str, job=None) -> Path | None:
    """Write everything knowable about ``instance`` to a file. Returns its path.

    Called from the delete path, so ordering is the whole contract: read from Nova
    first, write, and only then let the caller delete. Returns ``None`` if anything at
    all went wrong, which the caller ignores by design.
    """
    if directory is None:
        return None
    try:
        stamp = datetime.now(timezone.utc)
        details = fleet.server_details(conn, instance.id)
        actions = fleet.server_actions(conn, instance.id)
        console = fleet.console_log(conn, instance.id)

        directory.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        name = f"{stamp:%Y%m%dT%H%M%SZ}-{instance.name}-{instance.id[:8]}.txt"
        path = directory / name

        body = _render(stamp, instance, why, job, details, actions, console)
        # os.open with the mode up front, rather than write-then-chmod: the file must
        # never exist world-readable, not even for the width of one syscall.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
        with os.fdopen(fd, "w") as fh:
            fh.write(body)
    except Exception:
        logger.warning(
            "could not capture diagnostics for %s; deleting it anyway",
            getattr(instance, "id", "?"),
            exc_info=True,
        )
        return None

    logger.warning(
        "wrote pre-delete diagnostics for %s to %s (%d bytes, console %s)",
        instance.name,
        path,
        len(body),
        "captured" if console else "UNAVAILABLE",
    )
    return path


def _render(stamp, instance, why, job, details, actions, console) -> str:
    out = [
        "# sivacor-autoscaler pre-delete diagnostics",
        f"captured:    {stamp.isoformat()}",
        f"instance:    {instance.name} ({instance.id})",
        f"status:      {instance.status}",
        f"created_at:  {instance.created_at}",
        f"reap reason: {why}",
        "",
    ]

    if job is not None:
        out += [
            "## submission still RUNNING on this instance",
            f"job id:         {job.id}",
            f"last heartbeat: {job.heartbeat}",
            "",
            "The gap between that heartbeat and the poweroff is the diagnosis. ~17 min",
            "means the worker's idle supervisor powered the VM off after 8 unreachable",
            "ticks -- it only reaches that branch once `docker ps` has succeeded AND",
            "shown zero analysis containers, so the run's container was already gone.",
            "Look below for oom-kill, 'No space left', or a dockerd crash just before",
            "the first UNREACHABLE line.",
            "",
        ]
    else:
        out += [
            "## no running submission recorded for this instance",
            "Either it finished cleanly, or its job document was deleted (which erases",
            "meta.worker_queue, the only join between a job and the VM that ran it).",
            "",
        ]

    out.append("## nova server actions")
    if actions:
        out += [
            "A guest that ran `systemctl poweroff` leaves NO action here; an API stop",
            "does. If the only action is 'create', the VM shut itself down.",
            "",
        ]
        for a in actions:
            out.append(
                f"  {a.get('start_time')}  {a.get('action')}  "
                f"message={a.get('message')!r}"
            )
    else:
        out.append("  (none readable)")
    out.append("")

    out.append("## nova server details (user_data redacted)")
    out.append(_json(details))
    out.append("")

    out.append("## serial console buffer")
    out.append(
        console
        if console
        else "  (unavailable -- Nova returned nothing, or the instance was already gone)"
    )
    out.append("")
    return "\n".join(out)


def _json(value) -> str:
    try:
        return json.dumps(value, indent=2, sort_keys=True, default=str)
    except Exception:
        logger.warning("could not serialise server details", exc_info=True)
        return repr(value)
