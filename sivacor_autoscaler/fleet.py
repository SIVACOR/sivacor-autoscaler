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


#: Which catalogue rung an instance is, as a tag: ``sivacor-size:60``.
#:
#: A *third* tag is safe because :func:`list_fleet` filters on tag **subset** -- it
#: requires :data:`FLEET_TAG` and the deployment tag to be present and ignores anything
#: else -- so instances predating this carry two tags and are matched exactly as before.
#: Their size reads as ``None``, which :func:`plan._instance_rung` resolves to the
#: cheapest rung.
#:
#: Tagged rather than kept in metadata because ``list_fleet`` already reads tags for
#: every instance on every tick, so this costs no extra call. The human-facing copy goes
#: to metadata alongside the existing role/deployment properties, which is what
#: ``openstack server show`` prints.
SIZE_TAG_PREFIX = "sivacor-size:"


def size_tag(memory_gb: int) -> str:
    return f"{SIZE_TAG_PREFIX}{memory_gb}"


def _flavor_name(server) -> str | None:
    """The flavour's name from a detailed server listing, or ``None``.

    Nova returns the flavour as a nested object whose name lives in ``original_name``;
    ``name`` on some microversions; and the whole attribute can be absent. None of those
    are worth an exception when the size tag is the primary source.
    """
    flavor = getattr(server, "flavor", None)
    if flavor is None:
        return None
    if isinstance(flavor, dict):
        return flavor.get("original_name") or flavor.get("name")
    return getattr(flavor, "original_name", None) or getattr(flavor, "name", None)


def _size_from(tags, flavor_name, flavor_sizes) -> int | None:
    """Which rung an instance is: its tag, else its flavour name, else unknown.

    The flavour fallback exists because it costs nothing -- ``servers(details=True)``
    already returns the flavour and :func:`list_fleet` used to discard it -- and it
    covers the one case the tag cannot: an instance booted before this tag existed, or
    by hand. ``None`` is a legitimate answer and callers must handle it; see
    :attr:`plan.Instance.size`.
    """
    for tag in tags:
        if tag.startswith(SIZE_TAG_PREFIX):
            try:
                return int(tag[len(SIZE_TAG_PREFIX) :])
            except ValueError:
                logger.warning("ignoring unparseable size tag %r", tag)
    return flavor_sizes.get(flavor_name)

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
    worker_size: int | None = None,
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
    if worker_size is not None:
        # Informational on the box: nothing in the worker's startup branches on it. The
        # flavour already decided the hardware, and the container's memory cap is
        # derived from what the kernel reports rather than from anything told to it
        # (`lib.py::container_memory_limit`), which is why the OOM message can quote a
        # real `mem_limit_bytes`. This is here so `openstack console log show` and the
        # worker's own journal say which rung the controller believed it was booting --
        # the one place that belief and the hardware could silently disagree.
        lines.append(f"SIVACOR_WORKER_SIZE={shlex.quote(str(worker_size))}")
    return text.replace(INJECT_MARKER, "\n".join(lines))


def list_fleet(
    conn, deployment: str, flavor_sizes: dict[str, int] | None = None
) -> tuple[Instance, ...]:
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
    # flavour name -> rung, so an instance with no size tag can still be placed. Empty
    # means "do not try", which is every caller before P3.
    flavor_sizes = flavor_sizes or {}
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
                # `s.flavor` is an object on a detailed listing and `original_name` is
                # the flavour's name, which is what the catalogue keys on. Guarded at
                # every step: the attribute is absent on older microversions, and the
                # size tag is the primary source anyway -- a missing flavour here costs
                # a fallback, not a listing.
                size=_size_from(tags, _flavor_name(s), flavor_sizes),
            )
        )
    return tuple(out)


def create_instance(
    conn, cfg, user_data: str, flavor: str | None = None, size: int | None = None
) -> str:
    """Boot one worker. Returns its id.

    ``flavor`` overrides ``cfg.flavor`` for this one instance -- that is what makes the
    fleet heterogeneous -- and ``size`` is the catalogue rung it corresponds to, written
    as a tag so :func:`list_fleet` can read it back without a flavour lookup. Both
    default to the pre-P3 behaviour: one configured flavour, no size tag.

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
    want_flavor = flavor or cfg.flavor
    kwargs = {
        "name": name,
        "image_id": _require(conn.compute.find_image, cfg.image, "image"),
        "flavor_id": _require(conn.compute.find_flavor, want_flavor, "flavor"),
        "networks": [{"uuid": _require(conn.network.find_network, cfg.network, "net")}],
        # Base64 because compute.create_server passes user_data through verbatim --
        # only the higher-level conn.create_server() encodes, and that one would
        # allocate a floating IP, which workers must not have.
        "user_data": encoded,
        # Both tags, always. list_fleet() requires both, so an instance created with
        # only FLEET_TAG would be invisible to its own controller: never counted, never
        # reaped, and holding a quota slot until a human noticed.
        "tags": [FLEET_TAG, deployment_tag(cfg.deployment)]
        + ([size_tag(size)] if size is not None else []),
        # Metadata rather than tags for the human-facing copy: `openstack server show`
        # prints properties in full, which is where anyone debugging looks first.
        "metadata": {
            "sivacor_role": "worker",
            "sivacor_deployment": cfg.deployment,
            "sivacor_flavor": want_flavor,
            **({"sivacor_size_gb": str(size)} if size is not None else {}),
        },
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
    logger.info(
        "created %s (%s) as %s%s",
        name,
        server.id,
        want_flavor,
        f", size {size} GB" if size is not None else "",
    )
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
