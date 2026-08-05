"""User-data composition and error classification.

The quota tests matter more than they look: misclassifying a full allocation as a boot
failure trips the circuit breaker and stops the fleet scaling exactly when it is
busiest, and the only symptom is submissions queueing behind an idle controller.
"""

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
