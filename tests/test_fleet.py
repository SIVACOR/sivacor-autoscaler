"""User-data composition and error classification.

The quota tests matter more than they look: misclassifying a full allocation as a boot
failure trips the circuit breaker and stops the fleet scaling exactly when it is
busiest, and the only symptom is submissions queueing behind an idle controller.
"""

import base64
import gzip
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sivacor_autoscaler import fleet

TEMPLATE = Path(__file__).resolve().parents[2] / "deploy-sivacor" / "worker-cloud-init.sh"


def _build(**kw):
    return fleet.build_user_data(
        TEMPLATE,
        master_key_hex="deadbeef",
        redis_password="pw",
        manager_ip="10.3.37.197",
        **kw,
    )


@pytest.mark.skipif(not TEMPLATE.is_file(), reason="deploy-sivacor not checked out")
def test_injects_the_ephemeral_flag():
    """Without it the worker keeps consuming and the arithmetic stops holding."""
    assert "SIVACOR_EPHEMERAL_WORKER=1" in _build()


@pytest.mark.skipif(not TEMPLATE.is_file(), reason="deploy-sivacor not checked out")
def test_injects_secrets_and_manager_address():
    ud = _build()
    assert "MASTER_KEY_HEX=deadbeef" in ud
    assert "REDIS_PASSWORD=pw" in ud
    assert "MANAGER_TENANT_IP=10.3.37.197" in ud


@pytest.mark.skipif(not TEMPLATE.is_file(), reason="deploy-sivacor not checked out")
def test_worker_queues_are_the_templates_default_unless_asked():
    """Unset must leave `sivacor,<private queue>`, because the template is SHARED.

    One checkout is bind-mounted by the mirror and by production, which sit at
    different points of P2's rollout. If this injected anything by default, narrowing
    the queues for the armed deployment would narrow them for the flag-off one too --
    whose workers would then consume nothing while Girder kept publishing to `sivacor`,
    stranding every submission with the fleet reading healthy.
    """
    # Scoped to the injected block: the template mentions WORKER_QUEUES itself, which
    # is the point -- what must be absent is an injected assignment overriding it.
    injected = _build().split("# ---- injected")[1].split("\n\n")[0]
    assert "WORKER_QUEUES" not in injected


@pytest.mark.skipif(not TEMPLATE.is_file(), reason="deploy-sivacor not checked out")
def test_worker_queues_are_injected_when_set():
    """P2 rollout step 4, as a value rather than an edit to the shared template."""
    ud = _build(worker_queues="private")
    assert "WORKER_QUEUES=private" in ud


@pytest.mark.skipif(not TEMPLATE.is_file(), reason="deploy-sivacor not checked out")
def test_the_template_still_defaults_the_queue_list_itself():
    """The injected value has to reach a `${WORKER_QUEUES:-...}`, or it does nothing.

    Pins the contract between this repo and deploy-sivacor's script: the marker sits
    above the header, so an injected assignment is what the default falls back from.
    """
    text = TEMPLATE.read_text()
    assert 'WORKER_QUEUES="${WORKER_QUEUES:-}"' in text, "header must accept an injection"
    assert 'private)   WORKER_QUEUES="${WORKER_QUEUE}"' in text, "the rollout-step-4 value"
    assert 'WORKER_QUEUES="sivacor,${WORKER_QUEUE}"' in text, "unset keeps both queues"
    assert "--queues=${WORKER_QUEUES}" in text


@pytest.mark.skipif(not TEMPLATE.is_file(), reason="deploy-sivacor not checked out")
def test_no_prepull_is_injected():
    """Workers stay interchangeable; see the P2.1/P3 note in build_user_data."""
    injected = _build().split("# ---- injected")[1].split("\n\n")[0]
    assert "PREPULL_IMAGES" not in injected


def test_missing_marker_is_fatal(tmp_path):
    """
    A silent no-op here boots an instance with no credentials.

    It would come up, consume nothing, and look healthy -- so this must fail at
    composition time rather than produce a plausible-looking script.
    """
    bad = tmp_path / "t.sh"
    bad.write_text("#!/bin/bash\necho hi\n")
    with pytest.raises(RuntimeError, match="SIVACOR_INJECT"):
        fleet.build_user_data(
            bad, master_key_hex="a", redis_password="b", manager_ip="c"
        )


class _Err(Exception):
    def __init__(self, status_code, msg):
        super().__init__(msg)
        self.status_code = status_code


@pytest.mark.parametrize(
    "exc,expected",
    [
        (_Err(403, "Quota exceeded for instances"), True),
        (_Err(403, "Maximum number of ports exceeded"), True),
        (_Err(403, "Policy doesn't allow os_compute_api:servers:create"), False),
        (_Err(500, "Unexpected API Error"), False),
        (Exception("connection reset"), False),
    ],
)
def test_quota_errors_are_distinguished_from_real_failures(exc, expected):
    assert fleet._is_quota_error(exc) is expected


class _Server:
    def __init__(self, id, tags, status="ACTIVE", name=None):
        self.id, self.tags, self.status = id, tags, status
        self.name = name or id
        self.created_at = "2026-08-05T15:00:00Z"


class _Conn:
    """Just enough of an openstacksdk connection for listing and creating."""

    def __init__(self, servers=()):
        self.servers_ = list(servers)
        self.created = []
        self.compute = self
        self.network = self

    def servers(self, details=True):
        return list(self.servers_)

    def find_image(self, name, ignore_missing=True):
        return _Server("image-id", [])

    def find_flavor(self, name, ignore_missing=True):
        return _Server("flavor-id", [])

    def find_network(self, name, ignore_missing=True):
        return _Server("net-id", [])

    def create_server(self, **kwargs):
        self.created.append(kwargs)
        return _Server("new-id", [])


OURS = fleet.deployment_tag("test.sivacor.org")
THEIRS = fleet.deployment_tag("sivacor.org")


def test_only_this_deployments_instances_are_listed():
    """
    The whole point: production and the mirror share one OpenStack project.

    Seeing a foreign instance is not cosmetic -- it is counted as available capacity
    (its claim marker lives in the *other* deployment's Girder), so a queued submission
    gets no VM, and every foreign SHUTOFF one is reaped by whoever ticks first.
    """
    conn = _Conn(
        [
            _Server("mine", [fleet.FLEET_TAG, OURS]),
            _Server("theirs", [fleet.FLEET_TAG, THEIRS]),
            _Server("legacy", [fleet.FLEET_TAG]),
            _Server("unrelated", []),
        ]
    )

    assert [i.id for i in fleet.list_fleet(conn, "test.sivacor.org")] == ["mine"]


def test_a_created_instance_carries_both_tags():
    """
    With only FLEET_TAG it would be invisible to its own controller.

    Never counted, never reaped, holding a quota slot until a human noticed -- strictly
    worse than the cross-talk this scoping fixes.
    """
    conn = _Conn()
    cfg = SimpleNamespace(
        deployment="test.sivacor.org",
        image="img",
        flavor="m3.medium",
        network="net",
        key_name=None,
        security_groups=[],
    )

    fleet.create_instance(conn, cfg, "#!/bin/bash\n")
    tags = conn.created[0]["tags"]

    assert fleet.FLEET_TAG in tags and OURS in tags
    assert conn.created[0]["metadata"]["sivacor_deployment"] == "test.sivacor.org"


def test_timestamp_parsing_tolerates_z_suffix():
    parsed = fleet._parse_time("2026-08-01T12:00:00Z")
    assert parsed is not None and parsed.tzinfo is not None


def test_unparseable_timestamp_is_none_not_an_exception():
    """An undated instance must not be mistaken for an ancient one and reaped."""
    assert fleet._parse_time("not a date") is None


# --- user_data is gzipped (2026-08-12) --------------------------------------
# The template hit 98 % of Nova's 65535-byte base64 limit and a comment-heavy commit
# pushed it 65 bytes over, at which point create_instance raised on every attempt and
# the fleet could not make a worker at all. Compression turns the cliff into headroom.


def _cfg():
    return SimpleNamespace(
        deployment="test.sivacor.org",
        image="img",
        flavor="m3.medium",
        network="net",
        key_name=None,
        security_groups=[],
    )


def _sent_user_data(conn):
    (kwargs,) = conn.created
    return kwargs["user_data"]


def test_user_data_is_gzipped_and_round_trips():
    conn, cfg = _Conn(), _cfg()
    script = "#!/bin/bash\n" + "# padding that compresses well\n" * 200
    fleet.create_instance(conn, cfg, script)
    blob = base64.b64decode(_sent_user_data(conn))
    assert blob[:2] == b"\x1f\x8b", "cloud-init detects gzip by magic; this must be gzip"
    assert gzip.decompress(blob).decode() == script


def test_gzip_is_deterministic():
    """user_data lives in Nova's DB; identical input must not look like a change."""
    a, b = _Conn(), _Conn()
    fleet.create_instance(a, _cfg(), "#!/bin/bash\necho hi\n")
    fleet.create_instance(b, _cfg(), "#!/bin/bash\necho hi\n")
    assert _sent_user_data(a) == _sent_user_data(b)


def test_a_template_that_would_not_fit_uncompressed_now_does():
    """The 2026-08-12 outage, as an assertion."""
    conn, cfg = _Conn(), _cfg()
    script = "#!/bin/bash\n" + "# a long explanatory comment line, as this file has\n" * 1400
    assert len(base64.b64encode(script.encode())) > fleet.USER_DATA_LIMIT
    fleet.create_instance(conn, cfg, script)  # must not raise
    assert len(_sent_user_data(conn)) < fleet.USER_DATA_LIMIT


def test_the_limit_still_applies_to_the_compressed_size():
    conn, cfg = _Conn(), _cfg()
    # Incompressible: os.urandom defeats gzip, so this exceeds the limit even packed.
    script = base64.b64encode(os.urandom(80_000)).decode()
    with pytest.raises(RuntimeError, match="gzipped"):
        fleet.create_instance(conn, cfg, script)
