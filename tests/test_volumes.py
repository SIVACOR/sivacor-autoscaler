"""Scratch volumes: create, attach, reclaim, and never leak one.

C2 of ``development_notes/cinder_volumes_plan.md``. What is under test is almost
entirely the *failure* paths, because the happy path is three API calls and the
interesting question is what happens when one of them does not answer.

**Why leaks matter more here than anywhere else in this fleet.** The OpenStack project
has **10 volumes** in total, two of them permanently the two deployments' own data
volumes -- production's is 800 GB and holds the filesystem assetstore. So this feature
has eight, and an eight-submission leak exhausts the *count* quota, after which the
next thing refused a volume is the manager. Instances are far more forgiving: 25 of
them, and a stuck one is caught by the max-lifetime sweep.
"""

from datetime import datetime, timedelta, timezone

import pytest

from sivacor_autoscaler import fleet
from sivacor_autoscaler.controller import Config, Controller
from sivacor_autoscaler.plan import Limits

DEPLOYMENT = "test.sivacor.org"
NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


class _Volume:
    def __init__(self, id, name, attachments=()):
        self.id, self.name = id, name
        self.attachments = list(attachments)


class _Attachment:
    def __init__(self, volume_id):
        self.volume_id = volume_id


class _Server:
    """Shaped for the real ``list_fleet``: both tags, or it is invisible."""

    def __init__(self, id, status="ACTIVE", age_minutes=30):
        self.id = id
        self.name = f"sivacor-worker-{id}"
        self.status = status
        self.tags = [fleet.FLEET_TAG, fleet.deployment_tag(DEPLOYMENT)]
        self.created_at = (NOW - timedelta(minutes=age_minutes)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self.flavor = None


class _Cloud:
    """Enough of an openstacksdk connection to record what was asked of it."""

    def __init__(self, *, create_volume_error=None, attach_error=None, server_error=None):
        self.compute = self
        self.network = self
        self.block_storage = self
        self.volumes_ = []
        self.servers_list = []
        self.created_servers = []
        self.deleted_servers = []
        self.deleted_volumes = []
        self.attachments_made = []
        self.detached = []
        self._create_volume_error = create_volume_error
        self._attach_error = attach_error
        self._server_error = server_error
        self._next = 0

    # --- compute -----------------------------------------------------------
    def servers(self, details=True):
        return list(self.servers_list)

    def find_image(self, name, ignore_missing=True):
        return _Volume("image-id", "img")

    def find_flavor(self, name, ignore_missing=True):
        return _Volume(f"flavor-{name}", name)

    def find_network(self, name, ignore_missing=True):
        return _Volume("net-id", "net")

    def create_server(self, **kwargs):
        if self._server_error:
            raise self._server_error
        self.created_servers.append(kwargs)
        return _Volume("server-1", kwargs.get("name"))

    def delete_server(self, instance_id, ignore_missing=True):
        self.deleted_servers.append(instance_id)

    def create_volume_attachment(self, server, volumeId, **kwargs):
        if self._attach_error:
            raise self._attach_error
        self.attachments_made.append((server, volumeId, kwargs))

    def delete_volume_attachment(self, volume_id, server, ignore_missing=True):
        self.detached.append((volume_id, server))

    def volume_attachments(self, instance_id):
        return [_Attachment(v.id) for v in self.volumes_]

    # --- block storage -----------------------------------------------------
    def create_volume(self, name, size, description):
        if self._create_volume_error:
            raise self._create_volume_error
        self._next += 1
        vol = _Volume(f"vol-{self._next}", name)
        vol.description = description
        self.volumes_.append(vol)
        return vol

    def delete_volume(self, volume_id, ignore_missing=True):
        self.deleted_volumes.append(volume_id)
        self.volumes_ = [v for v in self.volumes_ if v.id != volume_id]

    def volumes(self):
        return list(self.volumes_)


class _Cursor:
    """Chainable like pymongo's, because ``signals.spent_instance_ids`` sorts."""

    def sort(self, *a, **kw):
        return self

    def limit(self, *a, **kw):
        return self

    def __iter__(self):
        return iter(())


class _Girder:
    """``db[collection]`` for the per-tick setting reads and the claim scan."""

    def __init__(self, volumes_enabled=False):
        self.volumes_enabled = volumes_enabled

    def __getitem__(self, name):
        return self

    def find_one(self, query):
        if query.get("key") == "sivacor.volumes_enabled":
            return {"value": self.volumes_enabled}
        # Everything else, including the arm flag, reads as absent = off.
        return None

    def find(self, query, projection=None):
        return _Cursor()

    def count_documents(self, query):
        return 0


class _Redis:
    """One submission waiting, so ``decide`` actually wants an instance.

    Depth rather than an unclaimed-submission document because the arm flag is off in
    these tests: C2 is the volume machinery and must work on both dispatch paths, and
    the depth path is the one that needs no Girder job at all.
    """

    def __init__(self, depth=1):
        self.depth = depth

    def llen(self, queue):
        return self.depth

    def keys(self, pattern):
        return []

    def get(self, key):
        return None


def _cfg(tmp_path, volume_size_gb=100, **kw):
    template = tmp_path / "cloud-init.sh"
    template.write_text("#!/bin/bash\n" + fleet.INJECT_MARKER + "\necho hi\n")
    return Config(
        template=template,
        manager_ip="10.0.0.1",
        master_key_hex="ab",
        redis_password="pw",
        deployment=DEPLOYMENT,
        volume_size_gb=volume_size_gb,
        limits=Limits(max_instances=5),
        **kw,
    )


# --- naming and scoping ----------------------------------------------------


def test_a_volume_name_is_unique_and_carries_its_deployment():
    """Unique, because two volumes created in one tick must not collide; scoped,
    because the mirror and production share one OpenStack project and must never
    reap each other's -- the same hazard ``deployment_tag`` exists for."""
    first = fleet.volume_name(DEPLOYMENT)
    second = fleet.volume_name(DEPLOYMENT)

    assert first != second
    assert fleet.is_worker_volume(first, DEPLOYMENT)
    assert not fleet.is_worker_volume(first, "sivacor.org")


@pytest.mark.parametrize(
    "name",
    [None, "", "docker", "sivacor", "sivacor-worker-volume", "some-users-volume"],
)
def test_nothing_else_is_mistaken_for_ours(name):
    """The consequence of a false positive is deleting somebody's data volume --
    including, in this project, the 800 GB one holding the assetstore."""
    assert fleet.is_worker_volume(name, DEPLOYMENT) is False


def test_the_deployments_own_data_volume_is_not_ours():
    """Named ``sivacor`` and ``docker`` on the real project. Explicit, because these
    are the two volumes a matching bug would actually destroy."""
    assert not fleet.is_worker_volume("sivacor", DEPLOYMENT)
    assert not fleet.is_worker_volume("docker", DEPLOYMENT)


# --- the boot contract ----------------------------------------------------


def test_the_volume_id_reaches_the_worker(tmp_path):
    """The boot block builds its device path from this, so without it the volume is
    attached and never mounted -- and the worker silently runs on the root disk."""
    template = tmp_path / "ci.sh"
    template.write_text(fleet.INJECT_MARKER)

    text = fleet.build_user_data(
        template,
        master_key_hex="ab",
        redis_password="pw",
        manager_ip="10.0.0.1",
        volume_id="f2ae50b4-9135-4b0e-9cb8-ad906ee7316f",
    )

    assert "SIVACOR_VOLUME_ID=f2ae50b4-9135-4b0e-9cb8-ad906ee7316f" in text


def test_no_volume_means_no_variable(tmp_path):
    """Absent, not empty-string: the boot block tests ``-n``, and an exported empty
    value is the same thing -- but a *missing* line is what every pre-C2 worker has,
    so this keeps the two identical."""
    template = tmp_path / "ci.sh"
    template.write_text(fleet.INJECT_MARKER)

    text = fleet.build_user_data(
        template, master_key_hex="ab", redis_password="pw", manager_ip="10.0.0.1"
    )

    assert "SIVACOR_VOLUME_ID" not in text


# --- attachment ------------------------------------------------------------


def test_an_attachment_is_set_to_die_with_its_instance():
    """``delete_on_termination`` is the backstop for the controller dying between
    deleting a server and deleting its volume -- a window that exists on every
    restart and deploy. Nova does NOT default it on: the C0.2 probe reported
    ``Delete On Termination: False``."""
    cloud = _Cloud()

    fleet.attach_volume(cloud, "server-1", "vol-1")

    server, volume_id, kwargs = cloud.attachments_made[0]
    assert (server, volume_id) == ("server-1", "vol-1")
    assert kwargs["delete_on_termination"] is True


# --- the create path, and its rollbacks -----------------------------------


def test_a_worker_gets_a_volume_when_armed(tmp_path):
    """The happy path, in the order that is forced rather than chosen: volume, then
    user-data carrying its id, then instance, then attach."""
    cloud = _Cloud()
    ctl = Controller(cloud, _Redis(), _Girder(volumes_enabled=True), _cfg(tmp_path))

    ctl.step()

    assert len(cloud.volumes_) == 1
    assert cloud.volumes_[0].name.startswith(fleet.VOLUME_NAME_PREFIX)
    assert len(cloud.created_servers) == 1
    # The instance was told which volume to mount.
    user_data = cloud.created_servers[0]["user_data"]
    assert cloud.volumes_[0].id in fleet.gzip.decompress(
        fleet.base64.b64decode(user_data)
    ).decode()
    assert cloud.attachments_made
    assert not cloud.deleted_volumes


def test_no_volume_is_created_while_the_setting_is_off(tmp_path):
    """Off is the default and must be byte-for-byte the pre-C2 path: not a volume of
    size zero, not an unattached one -- no Cinder call at all."""
    cloud = _Cloud()
    ctl = Controller(cloud, _Redis(), _Girder(volumes_enabled=False), _cfg(tmp_path))

    ctl.step()

    assert cloud.volumes_ == []
    assert cloud.attachments_made == []
    assert len(cloud.created_servers) == 1, "the fleet still boots workers"


def test_no_volume_is_created_when_no_size_is_configured(tmp_path):
    """Two independent switches. Armed in Girder but unsized here is a deployment
    that has not been told how big, and it must not guess."""
    cloud = _Cloud()
    ctl = Controller(
        cloud,
        _Redis(),
        _Girder(volumes_enabled=True),
        _cfg(tmp_path, volume_size_gb=None),
    )

    ctl.step()

    assert cloud.volumes_ == []
    assert len(cloud.created_servers) == 1


def test_a_volume_is_reclaimed_when_the_instance_cannot_be_created(tmp_path):
    """Otherwise it is an orphan only a sweep could find -- and C5's sweep does not
    exist yet, so this is the only thing standing between a Nova hiccup and a
    permanently held slot out of eight."""
    cloud = _Cloud(server_error=RuntimeError("nova said no"))
    ctl = Controller(cloud, _Redis(), _Girder(volumes_enabled=True), _cfg(tmp_path))

    ctl.step()  # must not raise: step classifies its own failures

    assert cloud.deleted_volumes, "the volume outlived the instance that never existed"
    assert cloud.volumes_ == []


def test_both_halves_go_back_when_the_attach_fails(tmp_path):
    """A worker holding an unattached volume is the worst outcome available: it has
    spent one of eight volumes AND will run on the root disk this plan exists to
    escape. The boot block would refuse to start anyway rather than guess a device."""
    cloud = _Cloud(attach_error=RuntimeError("cinder said no"))
    ctl = Controller(cloud, _Redis(), _Girder(volumes_enabled=True), _cfg(tmp_path))

    ctl.step()

    assert cloud.deleted_volumes
    assert cloud.deleted_servers, "a worker with no scratch disk must not be left running"


def test_a_full_cinder_quota_is_backpressure_not_a_breaker_trip(tmp_path):
    """The breaker exists to catch a broken image. Letting a full storage allocation
    feed it would stop the whole fleet precisely when it is busiest -- the S6 mistake
    in a new dimension, and the reason ``create_instance`` already treats Nova's
    quota this way."""
    cloud = _Cloud(create_volume_error=fleet.QuotaExceeded("over quota"))
    ctl = Controller(cloud, _Redis(), _Girder(volumes_enabled=True), _cfg(tmp_path))

    ctl.step()

    assert ctl.consecutive_failures == 0, "a full quota is not a creation failure"
    assert cloud.created_servers == [], "and nothing was booted without its disk"


# --- the reap path --------------------------------------------------------


def test_attachments_are_read_before_the_server_is_deleted(tmp_path):
    """An attachment is a property of the server, so deleting it first destroys the
    only cheap way to find what to reclaim -- the same ordering constraint the
    pre-delete diagnostics capture has."""
    cloud = _Cloud()
    cloud.volumes_ = [_Volume("vol-9", fleet.volume_name(DEPLOYMENT))]

    volumes = fleet.volumes_attached_to(cloud, "server-1")
    assert volumes == ("vol-9",)

    fleet.delete_volume(cloud, "vol-9", instance_id="server-1")
    assert cloud.detached == [("vol-9", "server-1")]
    assert cloud.deleted_volumes == ["vol-9"]


def test_a_listing_that_raises_does_not_stop_a_reap():
    """An instance left alive because its disks could not be enumerated is worse than
    a volume left behind: the instance burns SUs indefinitely, and the volume is what
    the C5 sweep is for."""

    class _Broken(_Cloud):
        def volume_attachments(self, instance_id):
            raise RuntimeError("nova unavailable")

    assert fleet.volumes_attached_to(_Broken(), "server-1") == ()


def test_a_failed_delete_is_loud_rather_than_silent(caplog):
    """This is exactly how the count quota leaks, so it cannot be swallowed."""

    class _Broken(_Cloud):
        def delete_volume(self, volume_id, ignore_missing=True):
            raise RuntimeError("cinder unavailable")

    fleet.delete_volume(_Broken(), "vol-1")

    assert "holds quota until reclaimed by hand" in caplog.text


# --- what the C5 sweep will build on --------------------------------------


def test_unattached_volumes_of_ours_are_identifiable():
    """The sweep's whole predicate. Attached means a worker is behind it; unattached
    means either the instance is gone or it never existed."""
    cloud = _Cloud()
    cloud.volumes_ = [
        _Volume("vol-1", fleet.volume_name(DEPLOYMENT), attachments=[{"id": "a"}]),
        _Volume("vol-2", fleet.volume_name(DEPLOYMENT)),
        _Volume("vol-3", "sivacor"),  # the assetstore volume, never ours
    ]

    found = dict(fleet.list_worker_volumes(cloud, DEPLOYMENT))

    assert found == {"vol-1": True, "vol-2": False}
    assert "vol-3" not in found


def test_another_deployments_volumes_are_not_listed():
    """Production must not sweep the mirror's, and vice versa."""
    cloud = _Cloud()
    cloud.volumes_ = [_Volume("vol-1", fleet.volume_name("sivacor.org"))]

    assert fleet.list_worker_volumes(cloud, DEPLOYMENT) == ()


# --- the reap gate --------------------------------------------------------


def test_reclaim_is_not_gated_on_the_live_setting(tmp_path):
    """Disarming must not leak every in-flight volume.

    An instance booted while armed still has a volume after someone turns the setting
    off, so the reap path keys off the *configured size* -- static -- rather than this
    tick's flag. Gating on the flag would start leaking one volume per live worker at
    the exact moment of a disarm, which is the worst time to begin.

    Driven through the real ``gather``/``decide`` rather than by monkeypatching them:
    the thing under test is an ordering inside ``step``, and a stubbed decision would
    assert my own belief about what ``decide`` returns for a SHUTOFF instance.
    """
    cloud = _Cloud()
    cloud.servers_list = [_Server("server-1", status="SHUTOFF")]
    cloud.volumes_ = [_Volume("vol-7", fleet.volume_name(DEPLOYMENT))]
    # Setting OFF, size still configured: the state right after a disarm.
    ctl = Controller(
        cloud, _Redis(depth=0), _Girder(volumes_enabled=False), _cfg(tmp_path)
    )

    ctl.step()

    assert cloud.deleted_servers == ["server-1"], "the SHUTOFF instance is reaped"
    assert "vol-7" in cloud.deleted_volumes, "and its volume goes with it"


def test_a_deployment_with_no_volumes_configured_pays_no_call(tmp_path):
    """The pre-C2 reap path, unchanged: no size configured means no attachment
    lookup, so a deployment that never uses this feature is not billed one extra
    Nova call per reap."""
    calls = []

    class _Counting(_Cloud):
        def volume_attachments(self, instance_id):
            calls.append(instance_id)
            return []

    cloud = _Counting()
    cloud.servers_list = [_Server("server-1", status="SHUTOFF")]
    ctl = Controller(
        cloud,
        _Redis(depth=0),
        _Girder(volumes_enabled=False),
        _cfg(tmp_path, volume_size_gb=None),
    )

    ctl.step()

    assert cloud.deleted_servers == ["server-1"]
    assert calls == []
