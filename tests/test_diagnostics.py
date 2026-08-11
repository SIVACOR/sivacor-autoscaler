"""A worker that dies mid-run must be described honestly and dumped before deletion.

The regression these cover is not an arithmetic one. Three production submissions were
reaped for "no heartbeat" on 2026-08-10/11; in each, the worker powered itself off
under a live run and the controller deleted it ~17 min later, logging "SHUTOFF, work
finished" -- taking the console buffer, the only record of *why*, with it. The fleet
behaviour was right; the reporting was wrong, and the delete was irreversible.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from sivacor_autoscaler import diagnostics
from sivacor_autoscaler.controller import Config, Controller
from sivacor_autoscaler.plan import FleetState, Instance, Limits, RunningJob, decide

NOW = datetime(2026, 8, 11, 5, 0, tzinfo=timezone.utc)
LIMITS = Limits(max_instances=5)


def inst(name, status="SHUTOFF", age=timedelta(hours=10)):
    return Instance(id=f"id-{name}", name=name, status=status, created_at=NOW - age)


def state(instances, running=None):
    return FleetState(
        queue_depth=0,
        serving=0,
        instances=tuple(instances),
        running_jobs=running or {},
        now=NOW,
    )


# --- how a reap is described ------------------------------------------------


def test_shutoff_with_no_running_job_is_a_clean_finish():
    d = decide(state([inst("w1")]), LIMITS)
    assert d.delete == ("id-w1",)
    assert d.alerts == ()
    assert d.abnormal == frozenset()
    assert "work finished" in d.reasons[0]


def test_shutoff_under_a_running_submission_is_an_alert_not_a_finish():
    """The exact production case: the VM is gone but Girder still has the job RUNNING."""
    running = {"id-w1": RunningJob(id="job-1", heartbeat=NOW - timedelta(minutes=17))}
    d = decide(state([inst("w1")], running), LIMITS)

    assert d.delete == ("id-w1",), "it must still be reaped -- it holds a quota slot"
    assert d.abnormal == frozenset({"id-w1"}), "but capture its logs first"
    (alert,) = d.alerts
    assert alert in d.reasons, "alerts is a severity view, not a second bucket"
    assert "job-1" in alert
    assert "0:17:00 ago" in alert
    assert "did NOT finish" in alert
    assert "work finished" not in alert


def test_heartbeat_of_a_different_awareness_does_not_break_the_reap():
    """pymongo hands back naive datetimes; OpenStack's are aware. Subtracting raises.

    A TypeError here would propagate out of decide() and stop the controller reaping
    anything at all -- trading a leaked instance for a better log line.
    """
    naive = datetime(2026, 8, 11, 4, 43)  # noqa: DTZ001 -- naive is the point here
    d = decide(state([inst("w1")], {"id-w1": RunningJob("job-1", naive)}), LIMITS)
    assert d.delete == ("id-w1",)
    assert "2026-08-11T04:43" in d.alerts[0]


def test_reap_reason_is_carried_by_id_not_parsed_back_out_of_the_log():
    d = decide(state([inst("w1"), inst("w2")]), LIMITS)
    assert set(d.reap_reasons) == {"id-w1", "id-w2"}


def test_lifetime_and_provisioning_reaps_are_abnormal_too():
    live = Instance(
        id="id-old", name="old", status="ACTIVE", created_at=NOW - timedelta(hours=40)
    )
    d = decide(state([live]), Limits(max_lifetime=timedelta(hours=30)))
    assert d.abnormal == frozenset({"id-old"})
    assert any("supervisor did not fire" in a for a in d.alerts)


# --- what gets written ------------------------------------------------------


class FakeConn:
    """Enough of openstacksdk for the capture path, with one read broken."""

    def __init__(self, console="BUSY\nUNREACHABLE (8/8)\nPOWERING OFF", explode=False):
        self.console = console
        self.explode = explode
        self.compute = self

    def get_server(self, _id):
        if self.explode:
            raise RuntimeError("nova is having a day")
        return _Server()

    def server_actions(self, _id):
        return [_Action()]

    def get_server_console_output(self, _id, length=None):
        if self.explode:
            raise RuntimeError("nova is having a day")
        return {"output": self.console}


class _Server:
    def to_dict(self):
        return {
            "id": "id-w1",
            "status": "SHUTOFF",
            "fault": None,
            # The reason REDACTED_SERVER_FIELDS exists: base64 user-data decodes to
            # MASTER_KEY_HEX and REDIS_PASSWORD in cleartext.
            "user_data": "TUFTVEVSX0tFWV9IRVg9ZGVhZGJlZWY=",
        }


class _Action:
    def to_dict(self):
        return {"action": "create", "start_time": "2026-08-10T19:07:12", "message": None}


def test_capture_writes_the_console_and_never_the_user_data(tmp_path):
    job = RunningJob(id="job-1", heartbeat=NOW - timedelta(minutes=17))
    path = diagnostics.capture(
        FakeConn(), tmp_path, inst("w1"), why="SHUTOFF mid-run", job=job
    )

    body = path.read_text()
    assert "POWERING OFF" in body
    assert "job-1" in body
    assert "TUFTVEVSX0tFWV9IRVg" not in body, "user_data must never reach a dump"
    assert '"user_data"' not in body, "not even as an empty key in the JSON"
    # 0600: a console buffer is uncurated output from a box handling research data.
    assert path.stat().st_mode & 0o777 == 0o600


def test_capture_survives_a_nova_that_will_not_answer(tmp_path):
    """A failed capture must still produce a file, and must never raise.

    The caller deletes the instance immediately afterwards; an exception here would
    leak it.
    """
    path = diagnostics.capture(FakeConn(explode=True), tmp_path, inst("w1"), why="x")
    assert path is not None
    assert "unavailable" in path.read_text()


def test_capture_into_an_unwritable_directory_returns_none_rather_than_raising():
    assert diagnostics.capture(FakeConn(), Path("/proc/nope"), inst("w1"), why="x") is None


# --- the controller does it in the right order ------------------------------


class FakeRedis:
    def llen(self, queue):
        return 0

    def keys(self, pattern):
        return []


class FakeCursor(list):
    def sort(self, *a):
        return self

    def limit(self, *a):
        return self


class FakeGirder:
    def __init__(self, jobs):
        self.jobs = jobs

    def __getitem__(self, name):
        return self

    def find(self, query, projection=None):
        return FakeCursor(self.jobs)

    def count_documents(self, query):
        return 0


class RecordingConn(FakeConn):
    """Records the order of console reads and deletes -- the contract that matters."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def servers(self, details=True):
        return [
            _FleetServer("id-w1", "w1", "SHUTOFF"),
        ]

    def get_server_console_output(self, _id, length=None):
        self.calls.append(("console", _id))
        return super().get_server_console_output(_id, length)

    def delete_server(self, instance_id, ignore_missing=True):
        self.calls.append(("delete", instance_id))


class _FleetServer:
    def __init__(self, id, name, status):
        self.id, self.name, self.status = id, name, status
        self.tags = ["sivacor-worker", "sivacor-deployment:test.sivacor.org"]
        self.created_at = "2026-08-10T19:07:12Z"


def test_console_is_read_before_the_instance_is_deleted(tmp_path):
    """Ordering is the entire value: Nova drops the buffer with the server."""
    conn = RecordingConn()
    jobs = [
        {
            "_id": "job-1",
            "status": 2,
            "meta": {"worker_queue": "sivacor.id-w1", "heartbeat": NOW},
        }
    ]
    cfg = Config(
        template=Path("/nonexistent"),
        manager_ip="10.0.0.1",
        master_key_hex="ab",
        redis_password="pw",
        deployment="test.sivacor.org",
        diagnostics_dir=tmp_path,
    )
    Controller(conn, FakeRedis(), FakeGirder(jobs), cfg).step()

    assert conn.calls == [("console", "id-w1"), ("delete", "id-w1")]
    assert len(list(tmp_path.iterdir())) == 1


def test_a_broken_diagnostics_query_still_reaps(tmp_path):
    """signals.running_jobs_by_instance swallows its errors, so the round proceeds."""

    class BrokenGirder(FakeGirder):
        def find(self, query, projection=None):
            if query.get("status") == 2:
                raise RuntimeError("mongo said no")
            return FakeCursor([])

    conn = RecordingConn()
    cfg = Config(
        template=Path("/nonexistent"),
        manager_ip="10.0.0.1",
        master_key_hex="ab",
        redis_password="pw",
        deployment="test.sivacor.org",
        diagnostics_dir=tmp_path,
    )
    Controller(conn, FakeRedis(), BrokenGirder([]), cfg).step()

    assert ("delete", "id-w1") in conn.calls
