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

import logging
from datetime import datetime, timedelta, timezone

import pytest

from sivacor_autoscaler import fleet
from sivacor_autoscaler.controller import Config, Controller
from sivacor_autoscaler.plan import Limits

DEPLOYMENT = "test.sivacor.org"
NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


class _Volume:
    def __init__(self, id, name, attachments=(), age_minutes=60):
        self.id, self.name = id, name
        self.attachments = list(attachments)
        # Cinder reports created_at with NO timezone -- verified on live JS2 -- which is
        # the shape that makes an aware/naive subtraction raise. The fake reproduces
        # that rather than the friendlier `Z` form Nova uses for servers.
        self.created_at = (NOW - timedelta(minutes=age_minutes)).strftime(
            "%Y-%m-%dT%H:%M:%S.000000"
        )


class _Attachment:
    def __init__(self, volume_id):
        self.volume_id = volume_id


class _Server:
    """Shaped for the real ``list_fleet``: both tags, or it is invisible."""

    def __init__(self, id, status="ACTIVE", age_minutes=30, volume_gb=None):
        self.id = id
        self.name = f"sivacor-worker-{id}"
        self.status = status
        self.tags = [fleet.FLEET_TAG, fleet.deployment_tag(DEPLOYMENT)]
        # From C3 the reap path keys off what the INSTANCE holds, read from this tag,
        # rather than off configuration -- so a test about reclaiming has to say the
        # instance had a volume.
        if volume_gb:
            self.tags.append(fleet.volume_gb_tag(volume_gb))
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

    def _reject_attach_while_building(self, *a, **kw):
        """What Nova really does, and what the unit tests originally did not.

        409 ``Cannot 'attach_volume' ... while it is in vm_state building``. Every
        create failed this way on the mirror 2026-08-21, and no test caught it because
        the fake accepted an attach at any time. It is here so nothing reintroduces a
        post-create attach.
        """
        raise RuntimeError(
            "ConflictException: 409: Cannot 'attach_volume' instance while it is in "
            "vm_state building"
        )

    def delete_server(self, instance_id, ignore_missing=True):
        self.deleted_servers.append(instance_id)

    def create_volume_attachment(self, server, volume=None, **kwargs):
        # Nothing should reach this any more: the volume rides along in the build.
        self.attachments_made.append((server, volume, kwargs))
        self._reject_attach_while_building()

    def delete_volume_attachment(self, server, volume, ignore_missing=True):
        # (server, volume), matching openstacksdk. The reversed call raised
        # `got multiple values for argument 'server'` on the mirror.
        self.detached.append((server, volume))

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


def _decide_to_create(monkeypatch, rung=None, disk_gb=100):
    """Make ``step`` execute one create of a given shape.

    Stubbing the decision rather than staging a Girder job on purpose: the controller's
    own docstring says every judgement lives in ``plan.decide`` and this module only
    does I/O, so a test of the *execution* should not have to reproduce the arithmetic
    to reach it. The arithmetic has its own tests below, against ``decide`` directly.

    ``rung=None`` by default -- an unsized create, which is what queue depth produces --
    because these tests are about the volume and a sized rung would need a catalogue
    loaded to resolve a flavour.
    """
    from sivacor_autoscaler import controller as controller_mod
    from sivacor_autoscaler.plan import Create, Decision

    monkeypatch.setattr(
        controller_mod,
        "decide",
        lambda state, limits: Decision(create=(Create(rung, disk_gb),)),
    )


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


def test_the_volume_rides_along_in_the_build(tmp_path, monkeypatch):
    """Attached by block device mapping, never by a later call.

    Nova refuses ``attach_volume`` while the instance is in ``vm_state building``, and
    ``create_server`` returns while it still is -- so a post-create attach fails 100 %
    of the time. Observed on the mirror 2026-08-21, 17 ms after the create, and missed
    by these tests until the fake started rejecting it the way Nova does.

    ``boot_index: -1`` means attach-but-do-not-boot; the root disk still comes from the
    image. And ``delete_on_termination`` lives here, which is also where it stops being
    Nova's default of False -- the C0.2 probe reported it as False on a plain attach.
    """
    cloud = _Cloud()
    _decide_to_create(monkeypatch)
    ctl = Controller(cloud, _Redis(), _Girder(volumes_enabled=True), _cfg(tmp_path))

    ctl.step()

    bdm = cloud.created_servers[0]["block_device_mapping"]
    # TWO entries. The boot entry is mandatory once a BDM is present at all: without it
    # Nova rejects the whole create with `400 Block Device Mapping is Invalid: Boot
    # sequence ... is not valid`, which is how the second mirror run failed. image_id
    # alone does not satisfy that check, and both are sent -- the same pair
    # python-openstackclient builds.
    assert len(bdm) == 2
    boot, scratch = bdm
    assert boot["boot_index"] == 0
    assert boot["source_type"] == "image"
    assert boot["destination_type"] == "local"
    assert boot["uuid"] == "image-id"
    assert cloud.created_servers[0]["image_id"] == "image-id", (
        "image_id is still sent alongside the BDM boot entry"
    )
    assert scratch["uuid"] == cloud.volumes_[0].id
    assert scratch["source_type"] == "volume"
    assert scratch["destination_type"] == "volume"
    assert scratch["delete_on_termination"] is True
    assert scratch["boot_index"] == -1, "must not displace the image as the boot disk"
    assert cloud.attachments_made == [], "no separate attach call: Nova would 409"


def test_a_block_device_mapping_always_carries_a_boot_entry(tmp_path, monkeypatch):
    """The invariant, asserted on its own because violating it fails the *create*.

    python-openstackclient refuses to send a BDM with no ``boot_index: 0`` entry at
    all, and Nova agrees: a mapping listing only the scratch volume is rejected
    outright, so the failure is a submission that never boots rather than a worker
    with no disk.
    """
    cloud = _Cloud()
    _decide_to_create(monkeypatch)
    ctl = Controller(cloud, _Redis(), _Girder(volumes_enabled=True), _cfg(tmp_path))

    ctl.step()

    bdm = cloud.created_servers[0]["block_device_mapping"]
    assert sum(1 for e in bdm if e.get("boot_index") == 0) == 1


def test_no_block_device_mapping_when_volumes_are_off(tmp_path, monkeypatch):
    """A create with no volume must be byte-for-byte the pre-C2 call -- an empty BDM
    list is not the same thing to Nova as an absent key."""
    cloud = _Cloud()
    _decide_to_create(monkeypatch)
    ctl = Controller(cloud, _Redis(), _Girder(volumes_enabled=False), _cfg(tmp_path))

    ctl.step()

    assert "block_device_mapping" not in cloud.created_servers[0]


# --- the create path, and its rollbacks -----------------------------------


def test_a_worker_gets_a_volume_when_armed(tmp_path, monkeypatch):
    """The happy path, in the order that is forced rather than chosen: volume, then
    user-data carrying its id, then instance, then attach."""
    cloud = _Cloud()
    _decide_to_create(monkeypatch)
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
    assert cloud.attachments_made == [], "attached via the build, not a second call"
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


def test_no_volume_when_the_submission_asked_for_none(tmp_path, monkeypatch):
    """The C3 semantic, and the one that keeps this feature cheap.

    C2 attached a fixed size to every worker; from C3 the size is the submission's, and
    a submission that asked for nothing gets **no Cinder call at all**. That is the
    common case -- the median workspace demand across the corpus is 1.32 GiB -- and it
    is why handing every production worker 100 GB of a shared 1000 GB quota was never
    the right shape.
    """
    cloud = _Cloud()
    _decide_to_create(monkeypatch, disk_gb=None)
    ctl = Controller(cloud, _Redis(), _Girder(volumes_enabled=True), _cfg(tmp_path))

    ctl.step()

    assert cloud.volumes_ == [], "asked for nothing, so nothing was created"
    assert "block_device_mapping" not in cloud.created_servers[0]
    assert len(cloud.created_servers) == 1, "the worker still boots"


def test_a_volume_is_reclaimed_when_the_instance_cannot_be_created(tmp_path, monkeypatch):
    """Otherwise it is an orphan only a sweep could find -- and C5's sweep does not
    exist yet, so this is the only thing standing between a Nova hiccup and a
    permanently held slot out of eight."""
    cloud = _Cloud(server_error=RuntimeError("nova said no"))
    _decide_to_create(monkeypatch)
    ctl = Controller(cloud, _Redis(), _Girder(volumes_enabled=True), _cfg(tmp_path))

    ctl.step()  # must not raise: step classifies its own failures

    assert cloud.deleted_volumes, "the volume outlived the instance that never existed"
    assert cloud.volumes_ == []


def test_a_full_cinder_quota_is_backpressure_not_a_breaker_trip(tmp_path, monkeypatch):
    """The breaker exists to catch a broken image. Letting a full storage allocation
    feed it would stop the whole fleet precisely when it is busiest -- the S6 mistake
    in a new dimension, and the reason ``create_instance`` already treats Nova's
    quota this way."""
    cloud = _Cloud(create_volume_error=fleet.QuotaExceeded("over quota"))
    _decide_to_create(monkeypatch)
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
    # (server, volume): the order openstacksdk actually takes. Reversed, it raises
    # `got multiple values for argument 'server'` -- which happened on the mirror and
    # was masked by delete_volume's own except, so the volume was still reclaimed and
    # only the logged traceback showed the detach had never run.
    assert cloud.detached == [("server-1", "vol-9")]
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

    found = {
        v.id: v.attached
        for v in fleet.list_worker_volumes(cloud, DEPLOYMENT, now=NOW)
    }

    assert found == {"vol-1": True, "vol-2": False}
    assert "vol-3" not in found, "the 800 GB assetstore volume must never be ours"


def test_another_deployments_volumes_are_not_listed():
    """Production must not sweep the mirror's, and vice versa."""
    cloud = _Cloud()
    cloud.volumes_ = [_Volume("vol-1", fleet.volume_name("sivacor.org"))]

    assert fleet.list_worker_volumes(cloud, DEPLOYMENT, now=NOW) == ()


# --- the reap gate --------------------------------------------------------


def test_the_reap_path_does_not_try_to_delete_the_volume(tmp_path, caplog):
    """It cannot work, and trying produces a false leak alarm on every reap.

    Once ``delete_instance`` has run the server is in ``task_state deleting``: Nova
    refuses the detach and Cinder refuses the delete of a still-attached volume. The
    old code then logged *"could not delete volume ... it now holds quota until
    reclaimed by hand"* for a volume Nova removed 26 s later -- every single reap,
    observed on the mirror 2026-08-21.

    **A false leak alarm is worse than no alarm.** The count quota is the one genuinely
    scary number in this feature, and an operator trained to ignore that line will
    ignore it when it is real.
    """
    caplog.set_level(logging.INFO)
    cloud = _Cloud()
    cloud.servers_list = [_Server("server-1", status="SHUTOFF", volume_gb=100)]
    cloud.volumes_ = [_Volume("vol-7", fleet.volume_name(DEPLOYMENT))]
    ctl = Controller(
        cloud, _Redis(depth=0), _Girder(volumes_enabled=False), _cfg(tmp_path)
    )

    ctl.step()

    assert cloud.deleted_servers == ["server-1"]
    assert cloud.deleted_volumes == [], "delete_on_termination reclaims it, not us"
    assert "holds quota until reclaimed by hand" not in caplog.text
    # Still named in the log, so a real leak stays cross-checkable.
    assert "vol-7" in caplog.text
    assert "rides out with" in caplog.text


def test_the_create_rollback_still_deletes_its_volume(tmp_path, monkeypatch):
    """The other caller, which genuinely holds an *unattached* volume.

    Dropping the delete from the reap path must not drop it here: this volume's
    instance never existed, so no termination hook can ever reclaim it.
    """
    cloud = _Cloud(server_error=RuntimeError("nova said no"))
    _decide_to_create(monkeypatch)
    ctl = Controller(cloud, _Redis(), _Girder(volumes_enabled=True), _cfg(tmp_path))

    ctl.step()

    assert cloud.deleted_volumes, "an orphan with no instance must still be deleted"


def test_reclaim_is_not_gated_on_the_live_setting(tmp_path, caplog):
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
    cloud.servers_list = [_Server("server-1", status="SHUTOFF", volume_gb=100)]
    cloud.volumes_ = [_Volume("vol-7", fleet.volume_name(DEPLOYMENT))]
    # Setting OFF, size still configured: the state right after a disarm.
    ctl = Controller(
        cloud, _Redis(depth=0), _Girder(volumes_enabled=False), _cfg(tmp_path)
    )

    caplog.set_level(logging.INFO)
    ctl.step()

    assert cloud.deleted_servers == ["server-1"], "the SHUTOFF instance is reaped"
    # The volume is reclaimed by delete_on_termination, so what this asserts is that
    # the attachment was still LOOKED UP after a disarm -- gating that lookup on the
    # live flag is what would lose track of in-flight volumes entirely.
    assert "vol-7" in caplog.text, "its volume is still accounted for"


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


# --- C3: Cinder as a third headroom dimension ------------------------------
#
# Against `decide` directly, per the plan's pure-function-first rule: with no volume
# limits configured this arithmetic must be byte-for-byte what it was, and the
# interesting cases are the stops -- a submission blocked by volumes or gigabytes must
# say WHICH, because S7 calls a submission waiting behind a quota it cannot name the
# least debuggable state this design can produce, and one naming the wrong limit worse.

from datetime import timedelta as _td

from sivacor_autoscaler.plan import (
    FleetState,
    Instance,
    SizeSpec,
    WaitingSubmission,
    decide,
)

LADDER = (SizeSpec(memory_gb=30, vcpus=8), SizeSpec(memory_gb=60, vcpus=16))


def _waiting(n, disk_gb, memory_gb=30):
    return tuple(
        WaitingSubmission(
            id=f"sub-{i}",
            age=_td(minutes=10 - i),
            assignable=True,
            memory_gb=memory_gb,
            disk_gb=disk_gb,
        )
        for i in range(n)
    )


def _armed(**kw):
    base = {"assign": True, "sizes": LADDER, "max_instances": 5}
    base.update(kw)
    return Limits(**base)


def _state(waiting=(), instances=(), spent=()):
    """``spent`` matters: an instance that is idle and matching gets ASSIGNED the
    waiting submission rather than provoking a create, so a test that wants an
    instance merely *holding quota* has to say it is already serving something."""
    return FleetState(
        queue_depth=0, serving=0, waiting=waiting, instances=instances, now=NOW,
        ready=frozenset(i.id for i in instances),
        spent=frozenset(spent),
    )


def test_the_disk_a_submission_asked_for_reaches_the_create():
    """C1 has recorded this since before anything could act on it."""
    d = decide(_state(waiting=_waiting(1, disk_gb=200)), _armed())

    assert [(c.rung, c.disk_gb) for c in d.create] == [(30, 200)]


def test_a_submission_that_asked_for_nothing_creates_no_volume():
    d = decide(_state(waiting=_waiting(1, disk_gb=None)), _armed())

    assert [(c.rung, c.disk_gb) for c in d.create] == [(30, None)]


def test_with_no_volume_limits_the_arithmetic_is_unchanged():
    """Production's branch until someone configures these, so it must not move."""
    waiting = _waiting(4, disk_gb=500)
    unlimited = decide(_state(waiting=waiting), _armed())

    assert len(unlimited.create) == 4, "no volume cap means volumes never block"
    assert not any("volume" in r for r in unlimited.reasons)


def test_the_volume_count_can_bind_before_the_instance_count():
    """The tighter of the two Cinder limits at this fleet's scale: 8 spare volumes
    against 5 instances means the count binds first only when it is set lower."""
    d = decide(_state(waiting=_waiting(3, disk_gb=10)), _armed(max_volumes=2))

    assert len(d.create) == 2
    assert any("volumes 2+1 > 2" in r for r in d.reasons), d.reasons


def test_gigabytes_can_bind_before_the_count():
    d = decide(_state(waiting=_waiting(3, disk_gb=400)), _armed(max_volume_gb=1000))

    assert len(d.create) == 2, "two 400 GB volumes fit in 1000, the third does not"
    assert any("volume GB 800+400 > 1000" in r for r in d.reasons), d.reasons


def test_a_volume_stop_names_volumes_rather_than_max_instances():
    """The `CAPPED` misattribution, in the new dimension.

    On 2026-08-20 a quota stop was reported as `max_instances=5`, sending the operator
    to raise a number that would change nothing. A volume stop must say volumes.
    """
    d = decide(_state(waiting=_waiting(2, disk_gb=10)), _armed(max_volumes=1))

    blocked = [r for r in d.reasons if "head of line" in r]
    assert blocked, d.reasons
    assert "volumes" in blocked[0]
    assert "max_instances" not in blocked[0]
    assert d.stopped_by == "quota" if hasattr(d, "stopped_by") else True


def test_the_head_of_line_message_names_the_disk_it_could_not_fit():
    d = decide(_state(waiting=_waiting(2, disk_gb=900)), _armed(max_volume_gb=1000))

    blocked = [r for r in d.reasons if "head of line" in r]
    assert "900 GB volume" in blocked[0], blocked


def test_a_submission_wanting_no_disk_is_never_blocked_by_a_volume_quota():
    """The ~90 % case must not queue behind the few that want disk.

    A full volume quota with nothing left would otherwise stall every ordinary
    submission on a deployment that had enabled the feature at all.
    """
    holding = (
        Instance(id="i1", name="w1", status="ACTIVE", size=30, volume_gb=1000,
                 created_at=NOW - _td(minutes=5)),
    )
    d = decide(
        _state(waiting=_waiting(2, disk_gb=None), instances=holding, spent=("i1",)),
        _armed(max_volume_gb=1000, max_volumes=1),
    )

    assert len(d.create) == 2, "no disk asked for, so the Cinder quota is irrelevant"


def test_a_volume_quota_stop_recovers_when_the_volume_comes_back():
    """The half that matters. A quota stop that never releases is indistinguishable
    from a stalled controller for the first twenty minutes."""
    holding = (
        Instance(id="i1", name="w1", status="ACTIVE", size=30, volume_gb=900,
                 created_at=NOW - _td(minutes=5)),
    )
    limits = _armed(max_volume_gb=1000)

    blocked = decide(
        _state(waiting=_waiting(1, disk_gb=500), instances=holding, spent=("i1",)), limits
    )
    assert blocked.create == (), "900 held + 500 wanted does not fit 1000"

    # ...the holder is reaped, its volume goes with it, and the next tick proceeds.
    freed = decide(_state(waiting=_waiting(1, disk_gb=500)), limits)
    assert len(freed.create) == 1, "the stop must release, not latch"


def test_an_instance_with_no_volume_tag_counts_as_zero():
    """Under-counts rather than over-counts: an instance booted before C3 has no tag,
    and inventing usage for it would build a quota wall out of nothing."""
    old = (
        Instance(id="i1", name="w1", status="ACTIVE", size=30,
                 created_at=NOW - _td(minutes=5)),
    )
    d = decide(
        _state(waiting=_waiting(1, disk_gb=1000), instances=old, spent=("i1",)),
        _armed(max_volume_gb=1000),
    )

    assert len(d.create) == 1


# --- C5.1: the orphan sweep ------------------------------------------------
#
# The only thing in the system that looks for a volume nothing is using. Since C2 the
# reap path deliberately leaves attached volumes to delete_on_termination -- attempting
# the delete produced a false leak alarm on every reap -- so if that flag ever fails to
# fire, this is what notices. The project has 10 volumes in total.

from sivacor_autoscaler.plan import WorkerVolume


def _vols(*specs):
    return tuple(
        WorkerVolume(id=i, attached=a, age=_td(minutes=m)) for i, a, m in specs
    )


def test_an_unattached_volume_past_the_grace_is_reclaimed():
    d = decide(
        FleetState(queue_depth=0, serving=0, now=NOW,
                   volumes=_vols(("vol-1", False, 30))),
        _armed(volume_orphan_grace=_td(minutes=15)),
    )

    assert d.delete_volumes == ("vol-1",)


def test_an_attached_volume_is_never_swept():
    """However old. An attached volume has a worker behind it, and a long-running
    submission is not an orphan -- the corpus has a 36.9-hour run in it."""
    d = decide(
        FleetState(queue_depth=0, serving=0, now=NOW,
                   volumes=_vols(("vol-1", True, 60 * 40))),
        _armed(volume_orphan_grace=_td(minutes=15)),
    )

    assert d.delete_volumes == ()


def test_a_volume_inside_the_grace_is_left_alone():
    """**The race the grace exists for, and it is not hypothetical.**

    The create path makes the volume *before* the instance -- it has to, because the
    boot block needs the volume id in its user-data -- so every healthy volume is
    briefly unattached. Measured on the mirror: created 16:39:21, attached by 16:40:02.
    A sweep with no grace would delete volumes that are about to be used, turning a
    safety net into the very leak-plus-failure it exists to prevent.
    """
    d = decide(
        FleetState(queue_depth=0, serving=0, now=NOW,
                   volumes=_vols(("vol-1", False, 1))),
        _armed(volume_orphan_grace=_td(minutes=15)),
    )

    assert d.delete_volumes == ()


def test_the_sweep_is_off_unless_a_grace_is_configured():
    """Off is the default, and off means nothing looks. Deliberate: a sweep that
    deletes is not something to enable by accident."""
    d = decide(
        FleetState(queue_depth=0, serving=0, now=NOW,
                   volumes=_vols(("vol-1", False, 999))),
        _armed(),
    )

    assert d.delete_volumes == ()


def test_a_reclaim_is_an_alert_not_a_log_line():
    """Every one of these is a leak that happened or a bug that caused one.

    A silent sweep is indistinguishable from a leak nobody noticed, which is the whole
    argument for the sweep existing.
    """
    d = decide(
        FleetState(queue_depth=0, serving=0, now=NOW,
                   volumes=_vols(("vol-1", False, 30))),
        _armed(volume_orphan_grace=_td(minutes=15)),
    )

    assert any("orphaned volume vol-1" in a for a in d.alerts), d.alerts
    assert any("delete_on_termination" in a for a in d.alerts), (
        "the message should name the mechanism that was supposed to handle it"
    )


def test_volume_ids_never_reach_the_instance_delete_list():
    """Separate fields on purpose: a volume id passed to delete_instance would ask Nova
    to delete a server that does not exist, and the reverse would be worse."""
    d = decide(
        FleetState(queue_depth=0, serving=0, now=NOW,
                   volumes=_vols(("vol-1", False, 30))),
        _armed(volume_orphan_grace=_td(minutes=15)),
    )

    assert d.delete == ()
    assert d.delete_volumes == ("vol-1",)


def test_the_sweep_runs_after_the_instance_deletes(tmp_path, monkeypatch):
    """Ordering, through step(). A volume whose instance was just reaped is reclaimed by
    delete_on_termination; sweeping first would race that and log a reclaim for
    something Nova was already handling. Anything still there next tick is real."""
    from sivacor_autoscaler import controller as controller_mod
    from sivacor_autoscaler.plan import Decision

    order = []
    cloud = _Cloud()
    cloud.servers_list = [_Server("server-1", status="SHUTOFF", volume_gb=10)]
    monkeypatch.setattr(
        controller_mod,
        "decide",
        lambda state, limits: Decision(
            delete=("server-1",), delete_volumes=("vol-orphan",)
        ),
    )
    monkeypatch.setattr(
        controller_mod.fleet, "delete_instance",
        lambda conn, i: order.append(("instance", i)),
    )
    monkeypatch.setattr(
        controller_mod.fleet, "delete_volume",
        lambda conn, v, instance_id=None: order.append(("volume", v)),
    )

    ctl = Controller(cloud, _Redis(depth=0), _Girder(), _cfg(tmp_path))
    ctl.step()

    assert order == [("instance", "server-1"), ("volume", "vol-orphan")]
