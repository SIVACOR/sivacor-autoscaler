"""OpenStack side: enumerate, create and delete worker instances.

Deliberately thin. Everything with judgement in it lives in :mod:`plan`; this module
only knows how to talk to Nova and how to compose user-data from the deployment's
cloud-init template.
"""

from __future__ import annotations

import base64
import logging
import re
import shlex
import uuid
from pathlib import Path

from .plan import Instance

logger = logging.getLogger(__name__)

#: Instances carry this tag so the fleet can be found without relying on names.
FLEET_TAG = "sivacor-worker"

INJECT_MARKER = "#__SIVACOR_INJECT__"

#: Nova's ceiling on the base64-encoded user_data blob.
USER_DATA_LIMIT = 65535


class QuotaExceeded(Exception):
    """Nova refused because the allocation is full.

    Distinct from a boot failure on purpose: a full allocation is **backpressure**,
    not an error. The submission stays queued, the researcher did nothing wrong, and
    it must not count towards the circuit breaker -- otherwise a busy fleet trips the
    breaker and stops scaling exactly when it is most wanted.
    """


def build_user_data(
    template: Path,
    *,
    master_key_hex: str,
    redis_password: str,
    manager_ip: str,
    girder_host: str | None = None,
    worker_image: str | None = None,
) -> str:
    """Inject configuration into the cloud-init template.

    The template lives in the deployment repo, not here: it is a property of the
    deployment, and duplicating it would let the two drift. A missing marker is fatal
    rather than a silent no-op -- an instance booted without credentials would come
    up, sit there consuming nothing, and look healthy.

    No prepull. Workers are interchangeable by design (P2.1/P3): a submission's images
    are pulled inside the run, where the heartbeat covers the silence, and that is what
    keeps ``desired == queue depth`` meaningful.
    """
    text = template.read_text()
    if INJECT_MARKER not in text:
        raise RuntimeError(f"{template} has no {INJECT_MARKER} line")
    lines = [
        "# ---- injected by sivacor-autoscaler ----",
        f"MASTER_KEY_HEX={shlex.quote(master_key_hex)}",
        f"REDIS_PASSWORD={shlex.quote(redis_password)}",
        f"MANAGER_TENANT_IP={shlex.quote(manager_ip)}",
        # Every instance serves one submission and stops. This is what makes the
        # worker drop its dispatch-queue consumer on accepting one (P3.2).
        "SIVACOR_EPHEMERAL_WORKER=1",
    ]
    if girder_host:
        lines.append(f"GIRDER_HOST={shlex.quote(girder_host)}")
    if worker_image:
        lines.append(f"WORKER_IMAGE={shlex.quote(worker_image)}")
    return text.replace(INJECT_MARKER, "\n".join(lines))


def list_fleet(conn) -> tuple[Instance, ...]:
    """Every worker instance, whatever its state.

    SHUTOFF ones matter as much as live ones: they are what the reap step exists for,
    and an unreaped instance holds a slot against the quota.
    """
    out = []
    for s in conn.compute.servers(details=True):
        if FLEET_TAG not in (s.tags or []):
            continue
        out.append(
            Instance(
                id=s.id,
                name=s.name or s.id,
                status=s.status or "UNKNOWN",
                created_at=_parse_time(s.created_at),
            )
        )
    return tuple(out)


def create_instance(conn, cfg, user_data: str) -> str:
    """Boot one worker. Returns its id.

    Raises :class:`QuotaExceeded` when the allocation is full so the caller can treat
    it as backpressure rather than failure.
    """
    encoded = base64.b64encode(user_data.encode()).decode()
    if len(encoded) > USER_DATA_LIMIT:
        raise RuntimeError(
            f"user_data is {len(encoded)} bytes encoded, over Nova's {USER_DATA_LIMIT}"
        )

    name = f"sivacor-worker-{uuid.uuid4().hex[:8]}"
    kwargs = {
        "name": name,
        "image_id": _require(conn.compute.find_image, cfg.image, "image"),
        "flavor_id": _require(conn.compute.find_flavor, cfg.flavor, "flavor"),
        "networks": [{"uuid": _require(conn.network.find_network, cfg.network, "net")}],
        # Base64 because compute.create_server passes user_data through verbatim --
        # only the higher-level conn.create_server() encodes, and that one would
        # allocate a floating IP, which workers must not have.
        "user_data": encoded,
        "tags": [FLEET_TAG],
        "metadata": {"sivacor_role": "worker"},
    }
    if cfg.key_name:
        kwargs["key_name"] = cfg.key_name
    if cfg.security_groups:
        kwargs["security_groups"] = [{"name": g} for g in cfg.security_groups]

    try:
        server = conn.compute.create_server(**kwargs)
    except Exception as exc:
        if _is_quota_error(exc):
            raise QuotaExceeded(str(exc)) from exc
        raise
    logger.info("created %s (%s)", name, server.id)
    return server.id


def delete_instance(conn, instance_id: str) -> None:
    conn.compute.delete_server(instance_id, ignore_missing=True)
    logger.info("deleted %s", instance_id)


def _require(finder, name, what):
    found = finder(name, ignore_missing=True)
    if found is None:
        raise RuntimeError(f"no {what} named {name!r}")
    return found.id


def _is_quota_error(exc) -> bool:
    """Nova signals a full allocation with 403 plus a quota message."""
    if getattr(exc, "status_code", None) != 403:
        return False
    return bool(re.search(r"quota|exceed", str(exc), re.IGNORECASE))


def _parse_time(value):
    if not value:
        return None
    if not isinstance(value, str):
        return value
    from datetime import datetime

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("could not parse instance timestamp %r", value)
        return None
