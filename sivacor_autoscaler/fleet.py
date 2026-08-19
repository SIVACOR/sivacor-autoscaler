"""OpenStack side: enumerate, create and delete worker instances.

Deliberately thin. Everything with judgement in it lives in :mod:`plan`; this module
only knows how to talk to Nova and how to compose user-data from the deployment's
cloud-init template.
"""

from __future__ import annotations

import base64
import gzip
import logging
import re
import shlex
import uuid
from pathlib import Path

from .plan import Instance

logger = logging.getLogger(__name__)

#: Instances carry this tag so the fleet can be found without relying on names.
FLEET_TAG = "sivacor-worker"

#: Prefix of the second, per-deployment tag. **Both tags are required** to consider an
#: instance ours.
#:
#: ``FLEET_TAG`` alone is not enough, and the reason is not hypothetical. Production
#: and the test mirror run their own controller against the **same OpenStack project**
#: (D3: one 25-instance allocation carries the manager, the mirror and any debug VM),
#: so a single shared tag makes each controller see the other's workers as its own:
#:
#: * ``spent`` and ``ready`` come from *this* deployment's Mongo and Redis, so a
#:   foreign worker is in neither -- it therefore counts as **available capacity**
#:   (:func:`plan.decide`), a queued submission gets no instance, and it stalls until
#:   the foreign VM disappears. Precisely run 6's phantom, arriving by cross-talk.
#: * every foreign ``SHUTOFF`` instance is reaped by whichever controller ticks first,
#:   and with D9 armed the foreign *live* ones are deleted at the deadline, since they
#:   can never appear in the local ready set.
#:
#: Kept out of :data:`FLEET_TAG` rather than folded into it so that an instance from a
#: deployment that predates this scoping is still recognisably fleet -- see
#: :func:`list_fleet`, which reports it rather than silently ignoring it.
DEPLOYMENT_TAG_PREFIX = "sivacor-deployment:"

INJECT_MARKER = "#__SIVACOR_INJECT__"

#: Foreign instance ids already reported, so the notice below fires once per instance
#: per process rather than every 30 s tick. Module state is ugly; a per-tick warning
#: for an 18-minute foreign worker is 36 identical lines, and silence is worse than
#: both -- an instance nothing will ever reap has to be visible somewhere.
_FOREIGN_REPORTED: set[str] = set()


def deployment_tag(deployment: str) -> str:
    """The per-deployment tag for ``deployment`` (typically the stack's ``domain``)."""
    return f"{DEPLOYMENT_TAG_PREFIX}{deployment}"

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
    worker_queues: str | None = None,
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
    if worker_queues:
        # What celery subscribes to. Unset leaves the template's own default,
        # `sivacor,<private queue>` -- both the shared dispatch queue and its own,
        # which is what every worker has always done. A deployment that has armed
        # targeted assignment sets it to just the private queue (P2 rollout step 4);
        # one that has not must not, and since both share one template file that
        # decision has to travel as a value rather than an edit.
        lines.append(f"WORKER_QUEUES={shlex.quote(worker_queues)}")
    return text.replace(INJECT_MARKER, "\n".join(lines))


def list_fleet(conn, deployment: str) -> tuple[Instance, ...]:
    """Every worker instance of ``deployment``, whatever its state.

    SHUTOFF ones matter as much as live ones: they are what the reap step exists for,
    and an unreaped instance holds a slot against the quota.

    **Filtering is on both tags**, for the reasons under
    :data:`DEPLOYMENT_TAG_PREFIX`. An instance carrying :data:`FLEET_TAG` but a
    different deployment tag -- or none, i.e. booted before this scoping existed -- is
    skipped and *reported*: this controller must not touch it, but something has to
    say so, because an instance no controller claims is one nothing will ever reap.
    """
    wanted = deployment_tag(deployment)
    out = []
    for s in conn.compute.servers(details=True):
        tags = set(s.tags or ())
        if FLEET_TAG not in tags:
            continue
        if wanted not in tags:
            if s.id not in _FOREIGN_REPORTED:
                _FOREIGN_REPORTED.add(s.id)
                foreign = sorted(t for t in tags if t.startswith(DEPLOYMENT_TAG_PREFIX))
                logger.info(
                    "ignoring %s (%s): tagged %s, not %s. Another deployment owns it "
                    "-- or nobody does, if that list is empty",
                    s.name or s.id,
                    s.id,
                    foreign or "no deployment",
                    wanted,
                )
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
    # GZIPPED, always. cloud-init sniffs the gzip magic and decompresses before
    # reading the script, so this is transparent to the template -- and it turns a
    # hard cliff into headroom: the 2026-08-12 template was 65,124 bytes base64
    # uncompressed (98 % of the limit, and 65 over once the injected block was added,
    # which stopped the fleet creating anything) against 25,324 compressed.
    #
    # Always, not "only when it would not fit". A path that runs solely in an
    # emergency is a path that has never been exercised when the emergency arrives;
    # the same argument retired the image-extraction fallback for py-spy.
    #
    # mtime=0 so identical input gives identical output -- user_data is stored in
    # Nova's DB and a gratuitously changing blob makes two instances look different
    # when they are not.
    encoded = base64.b64encode(gzip.compress(user_data.encode(), mtime=0)).decode()
    if len(encoded) > USER_DATA_LIMIT:
        raise RuntimeError(
            f"user_data is {len(encoded)} bytes gzipped+encoded, over Nova's "
            f"{USER_DATA_LIMIT}. It is already compressed, so the template "
            f"itself has to shrink -- see its SIZE BUDGET header."
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
        # Both tags, always. list_fleet() requires both, so an instance created with
        # only FLEET_TAG would be invisible to its own controller: never counted, never
        # reaped, and holding a quota slot until a human noticed.
        "tags": [FLEET_TAG, deployment_tag(cfg.deployment)],
        # Metadata rather than tags for the human-facing copy: `openstack server show`
        # prints properties in full, which is where anyone debugging looks first.
        "metadata": {"sivacor_role": "worker", "sivacor_deployment": cfg.deployment},
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


#: Fields of a Nova server that are never written to a diagnostics dump. ``user_data``
#: is the whole reason this list exists: it is base64, not encryption, and one decode
#: yields MASTER_KEY_HEX and REDIS_PASSWORD in cleartext (P2.2). The same mistake --
#: a debugging aid becoming a secret sink -- already reached a shared log on
#: 2026-08-01 via keystoneauth's request-body logging.
REDACTED_SERVER_FIELDS = ("user_data", "personality", "adminPass", "admin_password")


def server_details(conn, instance_id: str) -> dict | None:
    """Everything Nova knows about an instance, minus its credentials.

    ``fault`` is the field worth the round trip: when the hypervisor kills an
    instance, that is where it says so, and it is unreachable once the server is
    deleted. ``vm_state``/``power_state``/``task_state`` separate "guest powered
    itself off" from "someone called stop".
    """
    try:
        server = conn.compute.get_server(instance_id)
    except Exception:
        logger.warning("could not read details of %s", instance_id, exc_info=True)
        return None
    try:
        raw = server.to_dict()
    except Exception:
        logger.warning("could not serialise %s; falling back", instance_id, exc_info=True)
        raw = {k: getattr(server, k, None) for k in ("id", "name", "status", "fault")}
    return {k: v for k, v in raw.items() if k not in REDACTED_SERVER_FIELDS}


def server_actions(conn, instance_id: str) -> list[dict]:
    """Nova's action log for an instance: create, stop, reboot, delete.

    The point of asking is what is *absent*. A guest that ran ``systemctl poweroff``
    leaves no action, so a SHUTOFF instance whose log shows only ``create`` powered
    itself off; one showing ``stop`` was stopped through the API by something else.
    """
    try:
        return [a.to_dict() for a in conn.compute.server_actions(instance_id)]
    except Exception:
        logger.warning("could not read actions of %s", instance_id, exc_info=True)
        return []


def console_log(conn, instance_id: str, length: int | None = None) -> str | None:
    """The instance's serial console buffer, or ``None`` if it cannot be read.

    This is the only post-mortem a fleet worker has. It has no floating IP, so ssh is
    out even with a keypair; its journal dies with the disk; and the idle supervisor
    logs to ``journal+console`` (``worker-cloud-init.sh``) precisely so its
    BUSY/UNREACHABLE/POWERING OFF decisions land here. Nova drops the buffer with the
    server, so this must be read *before* :func:`delete_instance`, never after.
    """
    try:
        out = conn.compute.get_server_console_output(instance_id, length=length)
    except Exception:
        logger.warning("could not read console log of %s", instance_id, exc_info=True)
        return None
    if isinstance(out, dict):
        return out.get("output")
    return getattr(out, "output", None)


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
