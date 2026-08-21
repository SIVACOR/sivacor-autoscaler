"""What the fleet should look like, as a pure function.

All of the arithmetic and every guardrail lives here, deliberately free of Redis,
OpenStack and Girder, because this is the component whose bugs cost money: an
over-eager loop burns allocation, an under-eager one deadlocks submissions. Keeping
it pure means the interesting cases are cheap to test.

The caller (``controller``) gathers a :class:`FleetState`, asks :func:`decide` what to
do, and executes the answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import NamedTuple


@dataclass(frozen=True)
class SizeSpec:
    """One rung of the worker-size catalogue, as much of it as the arithmetic needs.

    The catalogue proper lives in Girder (``sivacor.worker_sizes``, P0.3) and carries a
    ``flavor`` name and a ``gated`` flag as well. Neither reaches here: S1 keeps the
    ``m3.*`` name server-side only, and ``gated`` guards the *picker* (P4), not the
    fleet. So this is deliberately the two numbers the quota arithmetic needs and
    nothing else -- if a third field ever seems necessary here, check first whether the
    decision really belongs in a pure function over quotas.
    """

    #: Advertised RAM in GiB. Both the enum value a submission asks for and the key
    #: everything here matches on -- S1's "the class *is* the number".
    memory_gb: int
    vcpus: int


@dataclass(frozen=True)
class Instance:
    """One worker VM, as much of it as the decision needs."""

    id: str
    name: str
    status: str
    created_at: datetime | None = None

    #: Which catalogue rung this instance *is*, by advertised RAM. From the
    #: ``sivacor-size:<n>`` tag written at create time, with the flavour name as a
    #: fallback (:func:`fleet.list_fleet`).
    #:
    #: ``None`` means neither could be resolved -- an instance booted before P3, or one
    #: whose flavour is not in the catalogue. Counted as the *smallest* rung for quota
    #: purposes, which under-counts rather than over-counts on purpose: over-counting
    #: invents a quota wall and stalls submissions, while under-counting merely lets
    #: Nova refuse the create, and ``QuotaExceeded`` already treats that as backpressure
    #: (S6 keeps it as the outer net for exactly this reason). It also means an
    #: unsized instance can only ever be matched to a smallest-rung submission, so it is
    #: never mistaken for capacity a large submission could use.
    size: int | None = None

    #: GB of scratch volume attached to this instance, or ``None`` for none.
    #:
    #: From the ``sivacor-volume-gb:<n>`` tag written at create time, the same channel
    #: :attr:`size` uses -- so counting what the fleet holds against the Cinder quota
    #: costs no API call beyond the server listing this already does. Deriving it from
    #: Cinder instead would be one extra round trip per tick to learn something the
    #: controller itself decided.
    #:
    #: ``None`` means an instance booted before C3, or one with no volume. Both count
    #: as zero, which under-counts rather than over-counts: over-counting invents a
    #: quota wall and stalls submissions, while under-counting merely lets Cinder
    #: refuse the create, and ``QuotaExceeded`` already treats that as backpressure.
    volume_gb: int | None = None

    @property
    def is_live(self) -> bool:
        """Booting or running, i.e. occupying an instance slot."""
        return self.status in ("BUILD", "ACTIVE")


@dataclass(frozen=True)
class RunningJob:
    """A submission Girder still believes is executing on a given instance.

    Carried purely so a reap can be *described* correctly. It changes no arithmetic:
    a SHUTOFF instance is deleted either way, because leaving it costs a quota slot
    and the submission is already unrecoverable -- its workspace died with the VM.
    What it changes is whether the operator can tell the two cases apart afterwards.
    """

    id: str
    #: ``meta.heartbeat``, the server's liveness signal. The gap between this and the
    #: poweroff is the whole diagnosis: ~17 min means the worker's idle supervisor
    #: powered the VM off after 8 unreachable ticks, i.e. celery died under a live run.
    heartbeat: datetime | None = None


@dataclass(frozen=True)
class WaitingSubmission:
    """A RUNNING submission that has not been given a worker yet.

    Absence of ``meta.worker_queue`` is the marker. Under targeted assignment (S2 of
    ``worker_sizing_plan.md``) the *controller* writes that field, so this set is both
    the assigner's input and, unchanged, the demand signal that already scales the
    fleet -- one query, one representation. See :attr:`FleetState.waiting`.

    Ages rather than timestamps because Girder stores ``created`` naive while ``now``
    here is tz-aware; :func:`signals.waiting_submissions` does the subtraction where
    that convention is known.
    """

    #: The Girder job id, as a string. Carried so an assignment can name it; the
    #: arithmetic uses it only as a stable tiebreaker.
    id: str
    age: timedelta

    #: GB of scratch volume this submission asked for, per ``meta.requested_disk_gb``.
    #:
    #: ``None`` means it asked for none, which is every submission on a deployment that
    #: has not enabled the feature and most submissions on one that has -- the median
    #: workspace demand measured across the corpus is 1.32 GiB. Distinct from ``0``
    #: because absent must stay the path with no Cinder call in it.
    disk_gb: int | None = None

    #: Advertised RAM the submission asked for, per ``meta.requested_memory_gb``.
    #:
    #: ``None`` means the submission predates P1's recording of it. Treated as the
    #: smallest catalogue rung, because that is the cheap direction: defaulting *up*
    #: silently multiplies the SU cost of the oldest submissions in the queue, and
    #: nobody reviews a bill for instances that all ran successfully. It is also only
    #: ever a guess -- what such a submission really ran on is whatever
    #: ``SIVACOR_OS_FLAVOR`` said at the time, which differs per deployment, so there is
    #: no right answer to recover here, only a safe one.
    memory_gb: int | None = None

    #: Whether this submission is *ours* to place, per ``meta.awaiting_assignment``.
    #:
    #: **Demand and assignability part company here, and only here.** Girder records
    #: which route each submission took at submit time, so flipping the arm flag while
    #: one is in flight cannot make the controller assign something ``submit_job``
    #: already published to the shared queue -- two workers on one workspace. During
    #: rollout step 3 both kinds are live at once: a dispatched-but-unclaimed
    #: submission is still real demand and still needs an instance created for it, it
    #: simply must not be *placed* by us.
    #:
    #: Defaults to ``True`` so the arithmetic tests stay about arithmetic;
    #: :func:`signals.waiting_submissions` maps a missing field to ``False``, which is
    #: the safe direction for a job document written before this existed.
    assignable: bool = True


@dataclass(frozen=True)
class Limits:
    #: Hard ceiling on worker instances. Configure it BELOW the OpenStack quota: the
    #: same 25 instances carry the manager, the test mirror and any hand-made debug
    #: VM, so deriving this from the quota guarantees a collision with ordinary work.
    #: 5-10 is right for the pilot; AEA volume is nowhere near the quota, and the
    #: value of one-VM-per-submission is scale-to-zero, not peak throughput.
    max_instances: int = 5

    #: Delete a live instance older than this no matter what it claims to be doing.
    #: A safety net against a self-shutdown supervisor that never fires. Must stay
    #: comfortably above the server's own max-runtime cap, or the controller will
    #: delete instances the reaper still considers healthy -- and every boot now
    #: includes a cold image pull, so this covers pull + run, not run alone.
    max_lifetime: timedelta = timedelta(hours=30)

    #: Stop creating after this many consecutive instances fail to register. Without
    #: it a broken image loops forever, burning ~8 SU/hr per stuck instance.
    breaker_threshold: int = 3

    #: How long an instance may take to announce readiness before it is written off
    #: as failed to provision: excluded from available capacity, then deleted (D9).
    #:
    #: **``None`` disables the check entirely, and that is the default on purpose.**
    #: Enabling it against a worker image that does not write readiness markers
    #: deletes every healthy instance the moment it passes the deadline -- the same
    #: rollout-ordering hazard as P4.5's tro-utils floor. Turn it on only once the
    #: deployed image is known to announce, and confirm with
    #: ``redis-cli KEYS 'sivacor:ready:*'`` after one worker has come up.
    #:
    #: Sizing: boot->ready has measured 122-140 s across five runs, so 10 minutes is
    #: roughly 4x margin. The asymmetry favours generosity -- deleting a healthy but
    #: slow instance costs one boot (~2.5 min) and the submission is re-created for,
    #: while keeping a dead one costs a stalled submission plus up to 30 h of SUs.
    provision_deadline: timedelta | None = None

    #: How long a RUNNING submission may sit unclaimed before it counts as demand in
    #: its own right, independently of queue depth. See :attr:`FleetState.waiting`.
    #:
    #: **Ignored entirely under :attr:`assign`**, where the race it skips cannot
    #: happen: nothing is published until an instance has been chosen, so a waiting
    #: submission is never a submission already being started.
    #:
    #: **The grace exists to skip one transient, not to be cautious.** A worker that
    #: has just reserved the head task is briefly both absent from the queue and
    #: unclaimed -- from the moment celery hands it the message to the moment
    #: ``prepare_submission`` calls ``claim()``, a second or two. Counting that would
    #: create an instance for a submission already being started, on every submit.
    #: Anything comfortably past it is a genuine stall: nothing legitimately takes
    #: minutes between reserving a message and claiming it, because ``claim()`` is
    #: deliberately the first thing that task does.
    #:
    #: Two minutes is ~4 controller rounds at the 30 s tick, so a stall is corrected
    #: in well under the 30 min the server-side reaper would take to fail the
    #: submission outright.
    unclaimed_grace: timedelta = timedelta(minutes=2)

    #: Arm targeted assignment: choose an instance per submission and publish the
    #: chain to that instance's private queue (S2/S3 of ``worker_sizing_plan.md``).
    #:
    #: **Off by default, and it must be flipped together with Girder's own flag, never
    #: alone.** With this on while ``submit_job`` still publishes to the shared queue,
    #: a submission is both dispatched and assigned -- two workers, one workspace.
    #: With it off while ``submit_job`` has stopped publishing, nothing ever reaches a
    #: worker and the fleet looks like a healthy idle system while every submission
    #: waits. That is why the flag is one value read by both processes rather than two
    #: settings that can disagree; see P2's rollout order.
    #:
    #: Arming it also makes :attr:`FleetState.ready` load-bearing -- an unready
    #: instance is never assigned to -- so the caller must populate ``ready`` whenever
    #: this is on, independently of :attr:`provision_deadline`.
    assign: bool = False

    #: How old an instance may be and still receive its *first* assignment.
    #:
    #: The hazard: an instance nobody has assigned anything to is idle for its whole
    #: life, so it powers itself off on its own schedule. Hand it work seconds before
    #: that and the chain lands in a queue whose consumer disappears -- and because
    #: ``meta.worker_queue`` is now set, the submission has left the waiting set and
    #: become demand nothing can see, until the server-side reaper fails it up to 30
    #: minutes later.
    #:
    #: **Sized against ``BOOT_GRACE_SEC``, not the idle timeout.** The supervisor
    #: refuses to power off before 600 s of uptime (``worker-cloud-init.sh:86``), which
    #: floors the earliest poweroff at 10 minutes even though the 300 s idle clock has
    #: long expired -- checked on a 2 min timer, so in practice 600-720 s. Eight minutes
    #: therefore stops assigning ~2 min before the first moment a poweroff is possible,
    #: and leaves a ~5.5 min window (11 ticks) from the 122-140 s boot->ready.
    #:
    #: Refusing is the cheap direction -- the submission stays waiting, so it still
    #: counts as demand and a fresh instance is created for it -- which is why a flat
    #: ceiling is enough. Aged-out instances are excluded from *capacity* as well, or
    #: the fleet deadlocks on capacity it will never assign to.
    assign_max_age: timedelta = timedelta(minutes=8)

    #: The worker-size catalogue, from Girder's ``sivacor.worker_sizes`` (P0.3).
    #:
    #: **Empty means "one shape, unknown", which is the pre-P3 behaviour exactly.** With
    #: no catalogue every submission and every instance collapses to a single anonymous
    #: rung, so the arithmetic below reduces term-for-term to what it was: size buckets
    #: of one, no vCPU/RAM checks, ``zip`` in the assigner. That equivalence is what lets
    #: this land unarmed, and :func:`decide`'s tests assert it.
    sizes: tuple[SizeSpec, ...] = ()

    #: OpenStack quota on vCPU and on RAM, in GiB. **``None`` disables each check, and
    #: that is the default on purpose** -- the same reasoning as
    #: :attr:`provision_deadline`. Enabling them against a fleet whose instances have no
    #: ``sivacor-size:`` tag makes every live instance count as the smallest rung, which
    #: *under*-counts usage and so cannot stall anything; but it also makes the numbers
    #: in the log wrong until the fleet has turned over, so turn them on knowing that.
    #:
    #: S6 supersedes D3 here: D3's "count instances, not vCPUs" holds only while every
    #: instance is the same shape. Read live 2026-08-13 the quota is 25 instances /
    #: 320 vCPU / 1220 GiB, so the instance count binds **only at the bottom rung** --
    #: at 125 GiB the RAM quota binds at nine, and a controller enforcing
    #: ``max_instances`` alone would take a Nova rejection instead, which is the very
    #: failure D3 wrote its rule to avoid.
    max_vcpus: int | None = None
    max_ram_gb: int | None = None

    #: Cinder volumes and gigabytes this fleet may hold. ``None`` = that check is off,
    #: which is the pre-C3 behaviour and correct for a deployment with no volumes.
    #:
    #: **A third headroom dimension after S6's two, and the tightest of the three.**
    #: Read live 2026-08-21: the project has 10 volumes / 2000 GB, of which two volumes
    #: and 1000 GB are the two deployments' own data volumes -- production's 800 GB one
    #: holds the filesystem assetstore. So the fleet has 8 volumes and 1000 GB, and with
    #: ``max_instances`` at 5 the **count** binds before the gigabytes do, leaving three
    #: spare as the entire margin for a leak.
    #:
    #: Set them **below** the real quota, for the same reason the other four are: the
    #: same allocation carries both deployments' data volumes, and deriving these from
    #: the quota guarantees a collision with the assetstore's ability to grow.
    max_volumes: int | None = None
    max_volume_gb: int | None = None


@dataclass(frozen=True)
class FleetState:
    #: Submissions published to the dispatch queue that no worker has taken yet.
    queue_depth: int
    #: Submissions currently executing. Reported for observability; the creation
    #: arithmetic uses :attr:`spent` instead -- see :func:`decide`.
    serving: int
    instances: tuple[Instance, ...] = ()
    #: Ids of instances that have already claimed a submission and therefore no
    #: longer consume the dispatch queue. Empty is safe: the arithmetic degrades to
    #: the older, stall-prone ``depth + serving`` estimate rather than misbehaving.
    #:
    #: Under :attr:`Limits.assign` the *controller* writes that marker, so this means
    #: "has been given a submission" rather than "has started one". The D9 skip below
    #: still rests on evidence that celery worked, but by a different route: only a
    #: :attr:`ready` instance is ever assigned to.
    spent: frozenset[str] = frozenset()
    #: Ids of instances whose celery worker started and reached the broker (D9).
    #: Consulted when :attr:`Limits.provision_deadline` is set -- see there for why
    #: that switch defaults to off -- **and** whenever :attr:`Limits.assign` is on,
    #: because an instance that has not registered is never assigned work.
    ready: frozenset[str] = frozenset()
    #: Every RUNNING submission that no worker has been given -- demand that
    #: :attr:`queue_depth` cannot see, because a reserved-but-unstarted message has
    #: already left the Redis list. Empty degrades to the depth-only reading, i.e. the
    #: behaviour before it existed. See :func:`signals.waiting_submissions`.
    #:
    #: This is *also* the assigner's work list under :attr:`Limits.assign`, and
    #: deliberately the same field: the demand signal and the assignment input are the
    #: same Mongo query, so representing them twice would let the fleet size itself
    #: from one reading and place work from another.
    waiting: tuple[WaitingSubmission, ...] = ()
    #: Consecutive instances that came up but never registered with celery.
    consecutive_failures: int = 0
    #: Instance id -> the submission Girder still has RUNNING on it. Diagnostics only;
    #: an empty mapping degrades every reap message to its old, less specific wording
    #: and nothing else. See :func:`signals.running_jobs_by_instance`.
    running_jobs: Mapping[str, RunningJob] = field(default_factory=dict)
    now: datetime | None = None


class Create(NamedTuple):
    """One instance to boot: which rung, and how much scratch disk it needs.

    **A pair rather than a bare rung, for the reason P3.1 made it a tuple rather than a
    count.** That note reads "create 3 no longer says what to create" once the fleet is
    heterogeneous in memory; it is heterogeneous in *disk* from C3 on, so a rung alone
    no longer says it either -- and the volume size is exactly what the Cinder headroom
    arithmetic needs per pending create.

    A ``NamedTuple`` so ``len(decision.create)`` and iteration keep working unchanged;
    only assertions that compared *contents* had to move.

    ``disk_gb`` is ``None`` for no volume, which is most submissions.
    """

    rung: int | None
    disk_gb: int | None = None


@dataclass(frozen=True)
class Decision:
    #: One entry per instance to boot, each a :class:`Create` -- the ``memory_gb`` rung
    #: it must be and the scratch disk it needs -- ordered
    #: oldest-submission-first (S7). A tuple rather than a count because the fleet is
    #: heterogeneous from P3 on: "create 3" no longer says what to create.
    #:
    #: ``None`` in place of a rung means "whatever :attr:`Limits.sizes` cannot tell us"
    #: -- demand read from queue *depth* rather than from a submission document, which
    #: carries no size. The caller boots its configured default for those. Only
    #: reachable while targeted assignment is off, i.e. on the shared-queue path.
    create: tuple[Create, ...] = ()
    delete: tuple[str, ...] = ()
    #: ``(submission id, instance id)`` pairs to bind, oldest submission first (S7).
    #: The caller claims each submission atomically and *then* publishes its chain to
    #: that instance's private queue -- never the reverse order, which can publish the
    #: same chain twice if the tick dies between the two. Empty unless
    #: :attr:`Limits.assign` is on.
    assign: tuple[tuple[str, str], ...] = ()
    #: Human-readable justification for every number above. Logged verbatim: this is
    #: the component whose decisions need to be auditable after the fact.
    reasons: tuple[str, ...] = field(default_factory=tuple)
    #: A **subset of** :attr:`reasons`: the ones describing a fleet that is not
    #: behaving as designed, and so deserving WARNING rather than INFO. A severity view
    #: rather than a separate bucket, so :attr:`reasons` stays the complete audit trail
    #: it is documented to be, and the normal chatter -- one "no new instances" line
    #: every 30 s -- still cannot bury an anomaly.
    alerts: tuple[str, ...] = field(default_factory=tuple)
    #: Ids whose deletion destroys evidence worth keeping: capture before deleting.
    #: A worker VM is the only place its own journal and console buffer exist, and
    #: ``delete_server`` is irreversible, so this is the last moment either can be read.
    abnormal: frozenset[str] = frozenset()
    #: Instance id -> why it is being reaped, the same text as in :attr:`reasons` or
    #: :attr:`alerts` without the ``reap <name>:`` prefix. Carried by id so the caller
    #: does not have to recover the association by parsing log lines back apart.
    reap_reasons: Mapping[str, str] = field(default_factory=dict)


def decide(state: FleetState, limits: Limits) -> Decision:
    """Return the create/delete actions the fleet needs.

    **Count available instances, not all live ones.** The tempting formula is
    ``create = depth - live``: never over-provision, since live instances will pick up
    the queue. That deadlocks. An ephemeral worker drops its consumer on the dispatch
    queue the moment it accepts a submission (P3.2), so a *spent* instance will never
    take another one -- and it powers off when done. With two spent instances and one
    queued submission, ``depth - live`` is ``-1``, the controller creates nothing, and
    nothing ever picks that submission up.

    The original fix estimated around it with ``desired = depth + serving``, which
    deadlocked less but still stalled: ``serving`` counts submissions *executing*, and
    an instance that has finished its submission but not yet powered off is neither
    serving nor available, yet still counts as ``live``. Measured 2026-08-01: a
    submission waited **4 min 45 s** behind exactly that, and would have waited ~13 min
    had a later submission not bumped the depth (D8, P3.5).

    So ask the question directly instead of estimating it::

        available = live instances that have NOT claimed a submission
        create    = depth - available          # clamped to the cap

    ``spent`` comes from a marker the worker writes to Girder when it claims a
    submission, not from a broker broadcast -- see :func:`signals.spent_instance_ids`
    for why that distinction matters.

    **Treat this as accurate, not exact.** A worker that becomes ready while two or
    more submissions are queued takes *two*: celery delivers them in one prefetch,
    milliseconds after startup and before ``cancel_consumer`` can run. The spare
    instance simply idles and is reaped, so this over-provisions slightly rather than
    stalling -- the safe direction. Nothing downstream may assume one submission per
    instance. (Under :attr:`Limits.assign` there is no shared queue to prefetch from,
    which is what makes one-per-instance true rather than hoped for.)

    **Deletes are never gated on the breaker.** A tripped breaker means "stop spending
    money", so continuing to reap is the whole point; skipping it would leak the very
    instances that tripped it.

    **Assignment is the same decision as capacity, which is why it lives here**
    (S3 of ``worker_sizing_plan.md``). Under :attr:`Limits.assign` the function also
    returns which waiting submission goes on which instance, oldest submission first
    (S7). Putting it anywhere else would mean implementing that ordering twice, from two
    snapshots taken at two different times, and the failure of the two disagreeing is
    over-provisioning -- silent, and paid for in SUs. Here it is answerable by a unit
    test with no OpenStack, no Mongo and no clock.

    Capacity and assignability are deliberately *different* sets. A booting instance is
    capacity, because it will serve; only an instance that has registered with the
    broker is assignable, because a chain published to a queue that never gets a
    consumer is a stall no signal can see. Instances past
    :attr:`Limits.assign_max_age` are excluded from both -- one that will never be
    assigned to must not absorb demand.

    **Sizes (P3).** Once :attr:`Limits.sizes` holds a catalogue *and* assignment is
    armed, both halves become per-size: a submission is only matched to an instance of
    the shape it asked for, and :attr:`Decision.create` says which shape each new
    instance must be. Scarce headroom is allocated strictly oldest-submission-first and
    **stops** at the first submission that does not fit, in both halves (S7).

    Two invariants hold this together, and both are asserted by the tests:

    * **With no catalogue, or with assignment off, this function is what it was.** The
      per-size path is behind that conjunction precisely so P3 can land while production
      still runs the shared queue -- the branch production takes is the code it already
      ran. Bucket-of-one reduces the matching to the ``zip`` it replaced and the
      allocation to ``clamp(demand - available, 0, headroom)``.
    * **Headroom is the minimum across three quotas, not just the instance count** (S6,
      superseding D3). :attr:`Limits.max_vcpus` and :attr:`Limits.max_ram_gb` default to
      ``None``, i.e. off, so enabling them is a deliberate act on a fleet whose instances
      carry size tags.
    """
    reasons: list[str] = []
    alerts: list[str] = []
    live = [i for i in state.instances if i.is_live]
    now = state.now or datetime.now(tz=_tz_of(live))

    # --- instances that never provisioned (D9) -----------------------------
    # A VM that boots but fails to provision never claims and never serves, so
    # without this it reads as available capacity forever -- and it cannot power
    # itself off either, because the supervisor is written by the script that
    # failed. Absence of a readiness marker past a boot deadline is what separates
    # that from "still booting" and from "healthy and idle".
    failed_to_provision: set[str] = set()
    if limits.provision_deadline is not None:
        for inst in live:
            # A spent instance is never written off, whatever the markers say: it holds
            # a submission, which is positive proof its celery worked -- by claiming it,
            # or by having been ready when the assigner chose it. The readiness write is
            # best-effort, so trusting it over that would be trusting the weaker signal.
            if inst.created_at is None or inst.id in state.ready or inst.id in state.spent:
                continue
            if now - inst.created_at > limits.provision_deadline:
                failed_to_provision.add(inst.id)

    # --- deletes -----------------------------------------------------------
    delete: list[str] = []
    seen: set[str] = set()
    abnormal: set[str] = set()
    reap_reasons: dict[str, str] = {}

    def _reap(inst, why: str, *, alarming: bool = False) -> None:
        # Guarded against double-listing: an instance can be both over the lifetime
        # ceiling and unprovisioned, and asking OpenStack to delete it twice turns a
        # tidy reap into a 404 on the second call.
        if inst.id in seen:
            return
        seen.add(inst.id)
        delete.append(inst.id)
        reap_reasons[inst.id] = why
        line = f"reap {inst.name}: {why}"
        reasons.append(line)
        if alarming:
            alerts.append(line)
            abnormal.add(inst.id)

    for inst in state.instances:
        if inst.status != "SHUTOFF":
            continue
        # "SHUTOFF, work finished" was this branch's only wording until 2026-08-11,
        # and it was an unchecked assumption. Three production submissions were reaped
        # for no heartbeat on 2026-08-10/11; in all three the worker had powered itself
        # off *mid-run* (celery unreachable for 8 ticks) and this loop deleted it ~17
        # min after the last heartbeat, logging "work finished" over the top. Since a
        # delete also destroys the console buffer and the journal, that one word was
        # the difference between a diagnosable failure and an unexplainable one.
        if job := state.running_jobs.get(inst.id):
            _reap(
                inst,
                f"SHUTOFF while submission {job.id} is still RUNNING"
                f"{_since(job.heartbeat, now)}. The worker did NOT finish its work: "
                "it powered off under a live run, so its own logs are the only record "
                "of why -- capture them before this delete",
                alarming=True,
            )
        else:
            _reap(inst, "SHUTOFF, work finished")
    for inst in live:
        if inst.id in failed_to_provision:
            _reap(
                inst,
                f"live for {now - inst.created_at} with no readiness marker, past "
                f"the {limits.provision_deadline} provisioning deadline; it can "
                "neither work nor reclaim itself",
                alarming=True,
            )
        if inst.created_at is None:
            continue
        age = now - inst.created_at
        if age > limits.max_lifetime:
            _reap(
                inst,
                f"live for {age}, over the {limits.max_lifetime} ceiling; "
                "supervisor did not fire",
                alarming=True,
            )

    # --- assignment (S2) ---------------------------------------------------
    # After the deletes, so a reaped instance is never handed work; before the creates,
    # because an instance nothing will be assigned to is not capacity either.
    assign: list[tuple[str, str]] = []
    aged_out: set[str] = set()
    if limits.assign:
        unassigned = [
            i
            for i in live
            if i.id not in state.spent
            and i.id not in failed_to_provision
            and i.id not in seen
        ]
        aged_out = {
            i.id
            for i in unassigned
            if i.created_at is not None
            and now - i.created_at > limits.assign_max_age
        }
        # Assignment is pessimistic where capacity is optimistic: a booting instance
        # will serve, but only a broker-registered one is given work, because a chain
        # published to an instance that never provisions sets `meta.worker_queue` -- so
        # the submission leaves the waiting set and becomes demand no signal can see.
        # That requirement is also what ties arming this to a worker image that
        # announces readiness.
        #
        # Fleet instances only: the manager's static worker announces readiness too but
        # is not an OpenStack server, so it never appears in `live`. A known gap for
        # D6's oversized-package tail, which wants `sivacor.static-01` as a target.
        assignable = [
            i for i in unassigned if i.id not in aged_out and i.id in state.ready
        ]
        # Youngest first: an unassigned instance has been idle its whole life, so the
        # youngest is furthest from its own poweroff. See assign_max_age.
        assignable.sort(key=_youngest_first(now))
        # Ours to place, per WaitingSubmission.assignable: during rollout step 3 a
        # submission Girder already dispatched is still demand, but publishing a second
        # chain for it would put two workers on one workspace.
        placeable = tuple(w for w in _oldest_first(state.waiting) if w.assignable)
        # Oldest submission first (S7), and only onto an instance of the shape it asked
        # for. With no catalogue every submission and every instance resolves to the same
        # anonymous rung, so this reduces exactly to the plain zip it replaced.
        pools: dict[int | None, list[Instance]] = {}
        for candidate in assignable:
            pools.setdefault(_instance_rung(candidate, limits), []).append(candidate)
        for sub in placeable:
            want = _submission_rung(sub, limits)
            pool = pools.get(want)
            if not pool:
                # **Stop, do not skip (S7).** Serving a smaller submission behind this
                # one raises utilisation and starves the large sizes indefinitely: under
                # a steady stream of small submissions the expensive one is always the
                # one that does not fit. It is also unexplainable to a researcher, who
                # would watch later submissions run while theirs waited.
                #
                # Only worth a line when instances were actually available and the wrong
                # shape. Running out of instances entirely is the ordinary scale-from-zero
                # case, already covered by the "none assignable" reason below.
                if any(pools.values()):
                    line = (
                        f"head of line: submission {sub.id} wants {want} GB and no "
                        f"instance of that size is free, so nothing behind it is "
                        f"assigned this tick (free: "
                        + ", ".join(
                            f"{k} GB x{len(v)}" for k, v in sorted(
                                pools.items(), key=lambda kv: (kv[0] is None, kv[0])
                            ) if v
                        )
                        + "). Creating one instead"
                    )
                    reasons.append(line)
                    alerts.append(line)
                break
            inst = pool.pop(0)
            assign.append((sub.id, inst.id))
            reasons.append(
                f"assign submission {sub.id} -> {inst.name}: waiting {sub.age}, "
                f"oldest of {len(placeable)} placeable ({len(state.waiting)} waiting)"
            )
        if aged_out and state.waiting:
            # Only worth saying in this combination. Ageing out with nothing waiting is
            # routine -- an over-provisioned instance idling towards its own poweroff --
            # but work waiting while an instance sits unassignable is never a busy
            # fleet.
            line = (
                f"{len(aged_out)} instance(s) too old to assign while "
                f"{len(state.waiting)} submission(s) wait: past "
                f"{limits.assign_max_age} they may power themselves off at any tick, "
                "so they are excluded from capacity too and replaced ("
                + ", ".join(sorted(i.name for i in live if i.id in aged_out))
                + ")"
            )
            reasons.append(line)
            alerts.append(line)
        elif placeable and not assignable:
            # Routine on an empty fleet -- the tick before the instances exist -- but
            # it is the only line that distinguishes that from the arming failure where
            # nothing is ever assignable because no worker announces readiness.
            reasons.append(
                f"{len(placeable)} submission(s) waiting, none assignable: "
                f"{len(live)} live, {len([i for i in live if i.id in state.spent])} "
                f"already assigned, {len([i for i in live if i.id in state.ready])} "
                f"registered with the broker"
            )
        elif state.waiting and not placeable:
            # The mixed-mode reading, and it must not look like the line above: these
            # submissions are not stuck, they are on the other path and a worker is
            # already coming for them. Only reachable during rollout step 3.
            reasons.append(
                f"{len(state.waiting)} submission(s) waiting, none of them ours to "
                "place: dispatched to the shared queue before assignment was armed"
            )

    # --- creates -----------------------------------------------------------
    # Instances remaining after this round's deletions still occupy their slots
    # until OpenStack actually removes them, so count conservatively against the cap.
    # `aged_out` (empty unless assignment is armed) must go here as well as out of
    # `assignable`: capacity nothing will ever be assigned to absorbs demand forever,
    # which is the run-3/run-5 stall through a new door.
    available = [
        i
        for i in live
        if i.id not in state.spent
        and i.id not in failed_to_provision
        and i.id not in aged_out
    ]
    # Demand is the *larger* of the two readings, never their sum: a submission
    # waiting in the queue is also RUNNING-and-unclaimed in Girder, so adding them
    # would double-count every ordinary submit and provision twice over. Taking the
    # max means depth still drives the normal case, while a submission depth has lost
    # sight of -- reserved by a worker that cannot start it -- still gets an instance.
    if limits.assign:
        # No grace: it exists to skip the reserve-to-claim window, which S2 removes
        # outright, and keeping it would delay every submission by two minutes. Still
        # max() against depth, because rollout step 3 runs both paths and a dispatched
        # submission is also unassigned.
        stalled = state.waiting
        demand = max(state.queue_depth, len(state.waiting))
    else:
        stalled = tuple(w for w in state.waiting if w.age > limits.unclaimed_grace)
        demand = max(state.queue_depth, len(stalled))
    if not limits.assign and len(stalled) > state.queue_depth:
        # The dispatch queue is not showing work that Girder says is unserved. An
        # anomaly *while submissions are dispatched blind*: either a worker is holding a
        # message it will not run, or a submission was published to a queue nobody
        # consumes. Hence the flag in the condition -- once assignment is armed the same
        # reading is the design, and would fire on every tick of a healthy fleet.
        # Appended to BOTH lists because `alerts` is documented as a severity view
        # *over* `reasons`, not a second bucket -- an alert missing from the audit trail
        # would be a bug.
        line = (
            f"{len(stalled)} submission(s) unclaimed for over {limits.unclaimed_grace} "
            f"but depth={state.queue_depth}: the dispatch queue has lost sight of "
            "work Girder still considers unserved; scaling on the unclaimed count"
        )
        reasons.append(line)
        alerts.append(line)
    headroom = limits.max_instances - len(live)
    if limits.assign and limits.sizes:
        # Per size, because an available instance of the *wrong shape* is not available
        # to this submission -- the one assumption the scalar formula below makes that
        # stops being true once the fleet is heterogeneous. Two 60 GB submissions and one
        # idle 30 GB instance is a shortfall of two, not one.
        wanted = _wanted_rungs(state, limits, available, stalled)
        shortfall = len(wanted)
        create_sizes, stopped_by = _allocate(
            wanted, live, limits, headroom, reasons, alerts
        )
    else:
        # Unarmed, or no catalogue: size information is absent end to end. Nothing is
        # placed by this controller, and every instance is the one shape
        # ``SIVACOR_OS_FLAVOR`` names -- so this is the pre-P3 arithmetic, term for term,
        # asking for unsized creates.
        #
        # **Keeping the two paths separate is what lets P3 land while production is
        # still flag-off**: the branch production takes is byte-for-byte the code it
        # already ran, so a regression there cannot be P3's. Same reasoning as the phase
        # order itself.
        shortfall = demand - len(available)
        # ``Create(None, None)`` and not a bare ``None``: same meaning as before C3 --
        # unsized, because depth carries no size, and no disk, because there is no
        # submission to have asked for one -- but the same type the armed path returns,
        # so the caller has one shape to consume. The *values* are what the plan means
        # by this branch being byte-for-byte the pre-P3 arithmetic.
        create_sizes = (Create(None, None),) * max(0, min(shortfall, headroom))
        # The scalar path can only ever be capped by the instance count.
        stopped_by = "instances" if len(create_sizes) < shortfall else None
    create = len(create_sizes)
    spent_live = len([i for i in live if i.id in state.spent])
    # Surfaced only when non-zero: on a healthy fleet it is noise on every tick, and
    # when it is non-zero it is the first thing worth seeing.
    dead = (
        f", {len(failed_to_provision)} unprovisioned" if failed_to_provision else ""
    )

    if state.consecutive_failures >= limits.breaker_threshold:
        if create:
            reasons.append(
                f"BREAKER OPEN: {state.consecutive_failures} consecutive instances "
                f"failed to register; refusing to create {create}. Deletes continue."
            )
        create_sizes = ()
        create = 0
    elif shortfall <= 0:
        reasons.append(
            f"no new instances: depth={state.queue_depth}{_unclaimed_note(stalled)}, "
            f"{len(available)} available of {len(live)} live ({spent_live} spent{dead}, "
            f"serving={state.serving})"
        )
    elif create < shortfall and stopped_by == "instances":
        # Not an error: the queue simply waits. Logged loudly because a cap that
        # silently throttles looks identical to a controller that has stopped working.
        reasons.append(
            f"CAPPED: want {shortfall} more but only {headroom} slot(s) left of "
            f"max_instances={limits.max_instances}; {shortfall - create} submission(s) "
            "will wait"
        )
    elif create < shortfall:
        # Something else stopped the allocation and has already said so in its own
        # words -- the quota, or a rung that left the catalogue. Naming max_instances
        # here as well would point the operator at a limit that is not binding.
        reasons.append(
            f"{shortfall - create} submission(s) will wait: see the head-of-line line "
            f"above ({headroom} of {limits.max_instances} instance slot(s) still free, "
            "so the instance cap is not what bound)"
        )
    else:
        reasons.append(
            f"create {create}: depth={state.queue_depth}{_unclaimed_note(stalled)}, "
            f"only {len(available)} available of {len(live)} live "
            f"({spent_live} spent{dead}, serving={state.serving})"
        )

    return Decision(
        create=create_sizes,
        delete=tuple(delete),
        assign=tuple(assign),
        reasons=tuple(reasons),
        alerts=tuple(alerts),
        abnormal=frozenset(abnormal),
        reap_reasons=reap_reasons,
    )



def _smallest_rung(limits) -> int | None:
    """The cheapest rung in the catalogue, or ``None`` when there is no catalogue.

    The fallback for everything whose size is unknown. Cheapest rather than largest on
    purpose: guessing *up* silently multiplies SU cost, and nobody reviews a bill for
    instances that all ran successfully.
    """
    return min((s.memory_gb for s in limits.sizes), default=None)


def _submission_rung(sub: WaitingSubmission, limits) -> int | None:
    """Which rung ``sub`` needs.

    A missing ``memory_gb`` means a pre-P1 submission and resolves to the smallest rung
    -- see :attr:`WaitingSubmission.memory_gb`. A value that is *not* in the catalogue is
    returned unchanged rather than rounded to something that exists: the catalogue
    shrinking under a queued submission is S1's "compatibility event", and the honest
    outcome is a visible head-of-line block naming the size it wants, not a silent
    downgrade onto hardware the run was never sized for. It then ages out through
    ``sivacor.assignment_timeout`` as ``reaped_no_worker``, which says the fleet is the
    problem -- which it is.
    """
    if not limits.sizes:
        return None
    return sub.memory_gb if sub.memory_gb is not None else _smallest_rung(limits)


def _instance_rung(inst: Instance, limits) -> int | None:
    """Which rung ``inst`` *is*, for matching and for quota arithmetic.

    Unlike :func:`_submission_rung`, an unrecognised size here **does** collapse to the
    smallest rung: see :attr:`Instance.size` for why under-counting an instance is the
    safe direction where under-serving a submission is not.
    """
    if not limits.sizes:
        return None
    known = {s.memory_gb for s in limits.sizes}
    return inst.size if inst.size in known else _smallest_rung(limits)


def _spec(limits, rung: int | None) -> SizeSpec | None:
    """The catalogue entry for ``rung``, or ``None`` if there isn't one."""
    for spec in limits.sizes:
        if spec.memory_gb == rung:
            return spec
    return None


def _quota_used(live, limits) -> tuple[int, int, int, int]:
    """What the live fleet already holds: vCPU, RAM (GiB), volumes, volume GB.

    All four from the instance listing the caller already has -- the two Cinder figures
    come from :attr:`Instance.volume_gb`, written as a tag at create time, so this stays
    a pure function over one snapshot. Asking Cinder instead would be a second source
    for something this controller decided itself, and two sources that can disagree is
    the failure S3 exists to avoid.
    """
    vcpus = ram = volumes = volume_gb = 0
    for inst in live:
        if spec := _spec(limits, _instance_rung(inst, limits)):
            vcpus += spec.vcpus
            ram += spec.memory_gb
        # Counted off the instance, not the catalogue: the volume size is per
        # submission from C3 on, so there is no rung to look it up from.
        if inst.volume_gb:
            volumes += 1
            volume_gb += inst.volume_gb
    return vcpus, ram, volumes, volume_gb


def _wanted_rungs(state, limits, available, stalled) -> list[int | None]:
    """One rung per unit of demand no existing instance can serve, oldest first (S7).

    The matching pass and the ordering are the same walk on purpose: doing them
    separately would mean deciding *how many* instances to create from one snapshot and
    *which submission each is for* from another, which is the two-owners failure S3
    exists to avoid, reproduced inside a single function.
    """
    pools: dict[int | None, int] = {}
    for inst in available:
        rung = _instance_rung(inst, limits)
        pools[rung] = pools.get(rung, 0) + 1
    out: list[Create] = []
    for sub in _oldest_first(stalled):
        rung = _submission_rung(sub, limits)
        if pools.get(rung):
            pools[rung] -= 1
        else:
            # Matching stays on the RUNG alone, deliberately: an existing instance can
            # serve this submission only if its memory matches, and its volume is
            # already sized and mounted. Matching on disk too would refuse a perfectly
            # good idle worker over a scratch disk nobody has asked to reuse -- and
            # under S2 every instance serves exactly one submission anyway, so a
            # mismatched volume cannot be inherited.
            out.append(Create(rung, sub.disk_gb))
    # Depth beyond what Girder accounts for: messages sitting in the shared queue, which
    # carry no size. Should be zero whenever assignment is armed -- nothing publishes
    # there -- but kept so this path still matches the ``max(depth, len(waiting))``
    # reading above during rollout step 3, and errs towards provisioning.
    # Depth carries neither size: no rung, and no disk. The caller boots its configured
    # defaults for both.
    out += [Create(None, None)] * max(0, state.queue_depth - len(stalled))
    return out


def _allocate(wanted, live, limits, headroom, reasons, alerts):
    """Turn wanted rungs into creates, oldest first, stopping at the first that will not
    fit (S7).

    **Strictly head-of-line: no skipping past a submission that does not fit.** Skipping
    raises utilisation and starves the large sizes forever, because the expensive
    submission is always the one that does not fit. See S7.

    Returns ``(rungs, stopped_by)`` where ``stopped_by`` is ``None``, ``"instances"``,
    ``"quota"`` or ``"catalogue"``. **The caller needs that to describe the cap
    correctly**, and getting it wrong is not cosmetic: observed on the mirror
    2026-08-20, a quota stop was reported as ``CAPPED: want 2 more but only 3 slot(s)
    left of max_instances=5``, which reads as "raise max_instances" when raising it
    would change nothing. A submission waiting behind a quota it cannot name is the
    least debuggable state this design can produce (S7); one waiting behind a quota
    that names the *wrong* limit is worse, because it sends the operator somewhere.
    """
    used_v, used_r, used_vol, used_vol_gb = _quota_used(live, limits)
    out: list[Create] = []
    for want in wanted:
        rung = want.rung
        if len(out) >= headroom:
            return tuple(out), "instances"
        spec = _spec(limits, rung)
        if spec is None and rung is not None:
            line = (
                f"head of line: a submission wants {rung} GB, which is not in the "
                f"catalogue ({', '.join(str(s.memory_gb) for s in limits.sizes)}). "
                "Nothing behind it is created this tick. The catalogue shrank under a "
                "queued submission; it will fail as reaped_no_worker unless the rung "
                "comes back"
            )
            reasons.append(line)
            alerts.append(line)
            return tuple(out), "catalogue"
        need_v = spec.vcpus if spec else 0
        need_r = spec.memory_gb if spec else 0
        need_disk = want.disk_gb or 0
        over = []
        if limits.max_vcpus is not None and used_v + need_v > limits.max_vcpus:
            over.append(f"vCPU {used_v}+{need_v} > {limits.max_vcpus}")
        if limits.max_ram_gb is not None and used_r + need_r > limits.max_ram_gb:
            over.append(f"RAM {used_r}+{need_r} > {limits.max_ram_gb} GB")
        # Cinder, and only when this create actually wants a volume: a submission
        # asking for no disk must never be blocked by a volume quota, or the ~90 % of
        # submissions that want nothing would queue behind the few that do.
        if need_disk:
            if limits.max_volumes is not None and used_vol + 1 > limits.max_volumes:
                over.append(f"volumes {used_vol}+1 > {limits.max_volumes}")
            if (
                limits.max_volume_gb is not None
                and used_vol_gb + need_disk > limits.max_volume_gb
            ):
                over.append(
                    f"volume GB {used_vol_gb}+{need_disk} > {limits.max_volume_gb}"
                )
        if over:
            line = (
                f"head of line: a {rung} GB instance"
                + (f" with a {need_disk} GB volume" if need_disk else "")
                + " does not fit the quota ("
                + "; ".join(over)
                + "), so nothing behind it is created this tick. Strictly oldest-first "
                "(S7): skipping to a smaller submission that does fit would starve this "
                "one indefinitely"
            )
            reasons.append(line)
            alerts.append(line)
            return tuple(out), "quota"
        out.append(want)
        used_v += need_v
        used_r += need_r
        if need_disk:
            used_vol += 1
            used_vol_gb += need_disk
    return tuple(out), None


def _oldest_first(waiting) -> tuple[WaitingSubmission, ...]:
    """Waiting submissions in the order S7 says they must be served.

    Oldest first, and the id as a tiebreaker so two submissions created in the same
    millisecond still order deterministically -- a decision that changes between ticks
    for no observable reason is one nobody can debug from the log.
    """
    return tuple(sorted(waiting, key=lambda w: (-w.age, w.id)))


def _youngest_first(now):
    """Sort key putting the instance with the most life left in it first.

    A worker's supervisor refuses to power off before ``BOOT_GRACE_SEC`` of *uptime*, so
    for an unassigned instance the time left is a direct function of its age: youngest
    is furthest from poweroff. See ``Limits.assign_max_age``.

    An undated instance sorts last rather than first. It was probably created moments
    ago, but it is also the one whose age cannot be checked against the ceiling, and
    preferring a candidate because less is known about it is the wrong instinct.
    """

    def key(inst):
        if inst.created_at is None:
            return (1, timedelta(0), inst.id)
        # Smallest age first, i.e. youngest first. The leading 0/1 is what keeps the
        # undated instance behind every dated one regardless of its age.
        return (0, now - inst.created_at, inst.id)

    return key


def _unclaimed_note(stalled) -> str:
    """``, unclaimed=2`` when submissions are stalled, nothing when none are.

    Omitted on a healthy fleet for the same reason as the ``unprovisioned`` note: a
    field that is always zero trains the reader to skip the line it appears on.
    """
    return f", unclaimed={len(stalled)}" if stalled else ""


def _since(heartbeat, now) -> str:
    """`` (last heartbeat 0:17:12 ago)``, or nothing if there is no heartbeat.

    Naive/aware mismatches are possible here in a way they are not for instance
    timestamps: ``meta.heartbeat`` comes back from pymongo naive by default, while
    ``now`` follows OpenStack and is aware. Subtracting those raises -- the same trap
    ``_tz_of`` exists for, and one that must not be allowed to break a reap.
    """
    if heartbeat is None:
        return ""
    try:
        return f" (last heartbeat {now - heartbeat} ago)"
    except TypeError:
        return f" (last heartbeat {heartbeat.isoformat()})"


def _tz_of(live):
    """Use an instance's own tzinfo so ages compare without a naive/aware TypeError.

    OpenStack hands back tz-aware timestamps; ``datetime.now()`` without a tz does
    not, and subtracting the two raises. The same trap the server-side reaper hit.
    """
    for inst in live:
        if inst.created_at is not None and inst.created_at.tzinfo is not None:
            return inst.created_at.tzinfo
    return None
