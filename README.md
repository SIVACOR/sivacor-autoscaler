# sivacor-autoscaler

Creates one OpenStack worker instance per queued SIVACOR submission and reaps it when
the work is done, so the allocation is only charged for instances that are actually
verifying something.

Runs on the manager, where the broker, Girder and the OpenStack credentials are.
Design rationale lives in `autoscaling_plan.md` (P3) in the workspace root; this README
covers only what is needed to run it.

## Shape

| module | role |
|---|---|
| `plan.py` | **all** arithmetic and guardrails, as a pure function. No I/O. |
| `signals.py` | queue depth (Redis `LLEN`) and running-submission count (Girder) |
| `fleet.py` | Nova: list / create / delete, and user-data composition |
| `controller.py` | the loop: gather → decide → execute |

The split is deliberate. This is the component whose bugs cost money in one direction
and stall submissions in the other, so the decision logic is pure and heavily tested
while the I/O around it stays dull.

## The arithmetic, and why `serving` is not optional

    desired = queue_depth + serving      # what the fleet must eventually cover
    create  = clamp(desired - live, 0, max_instances - live)

The obvious formula, `depth - live`, **deadlocks**. An ephemeral worker drops its
consumer on the shared dispatch queue the instant it accepts a submission, so a busy
instance will never take another — and it powers off when finished. With two busy
instances and one queued submission, `depth - live` is `-1`, nothing is created, and
that submission waits forever. Counting busy instances separately is what avoids it.

Treat the result as *accurate, not exact*: `cancel_consumer` is asynchronous and
messages are acked on receipt, so a worker occasionally takes a second submission. It
self-corrects, but nothing downstream may assume one submission per instance.

## Guardrails

- **Configured instance cap**, set *below* the OpenStack quota — the same quota covers
  the manager, the test mirror and any hand-made debug VM, so deriving the cap from it
  guarantees a collision. When the cap throttles, it says so: silent throttling is
  indistinguishable from a broken controller.
- **`403 QuotaExceeded` is backpressure, not failure.** Work stays queued, and it does
  **not** count towards the circuit breaker — otherwise a full allocation stops the
  fleet scaling exactly when it is busiest.
- **Circuit breaker** after N consecutive instances fail to register, so a broken image
  cannot loop and burn allocation. Deletes are never gated on it: a tripped breaker
  means "stop spending", and skipping reaps would leak the instances that tripped it.
- **Absolute maximum lifetime**, a net for a self-shutdown supervisor that never fires.
  Keep it above the server's own max-runtime cap, since every boot now includes a cold
  image pull.

## Running

```sh
pip install -e ".[test]"
source your-js2-app-cred.sh                  # or --cloud NAME for clouds.yaml

export SIVACOR_WORKER_TEMPLATE=../deploy-sivacor/worker-cloud-init.sh
export SIVACOR_MANAGER_TENANT_IP=10.3.37.197
export MASTER_KEY_HEX=...      REDIS_PASSWORD=...     # same values as the manager
export GIRDER_API_URL=https://girder.test.sivacor.org/api/v1
export GIRDER_API_KEY=...
export SIVACOR_OS_KEYPAIR=shakuras           # or workers launch with NO ssh key

sivacor-autoscaler --dry-run     # decide and explain, change nothing
sivacor-autoscaler --once        # one iteration
sivacor-autoscaler               # loop
```

`SIVACOR_OS_KEYPAIR` is optional and the startup log says which way it went. Leaving it
unset is a defensible production posture — workers hold no key material worth reaching
for — but a terrible debugging one, and a key cannot be added to a running instance.
Set it on any fleet you might need to inspect.

`--dry-run` runs the same `gather()` + `decide()` as a live round and skips only
execution, so it exercises the real code path rather than a parallel one. Start there.

## How instances actually get reclaimed

Two mechanisms, and the controller only owns the second.

**The worker powers itself off** (P3.3), via `sivacor-worker-idle-check` installed by
`deploy-sivacor/worker-cloud-init.sh` — not by this repo. Every 2 min it asks: past the
boot grace, no analysis containers, no active celery tasks, idle long enough? Then
`systemctl poweroff`. The controller reaps the resulting `SHUTOFF` instance, which is
why a worker needs no OpenStack credentials.

**That supervisor has three outcomes, not two**, and the third is what makes a reap
ambiguous. A probe that cannot see the docker socket is `blocked` — stay up, keep the
idle clock — because a supervisor that cannot see containers must not reclaim. But
celery failing to answer is `unreachable`, and 8 consecutive unreachable ticks (~17 min)
**power the box off**: a worker that cannot be reached over the broker also cannot be
*given* work. It was fail-safe-to-busy until 2026-08-01, when a half-open Redis socket
left a VM answering nothing and living to `SIVACOR_MAX_LIFETIME_HOURS`.

The cost of that change is the case this repo now has to report honestly: **an
`unreachable` poweroff is indistinguishable, from OpenStack, from a clean finish.** Both
arrive here as `SHUTOFF`. On 2026-08-10/11 three production submissions were reaped by
Girder for "no heartbeat" after exactly this — celery died under a live run, the VM
powered itself off ~17 min later, and the controller deleted it while logging
`SHUTOFF, work finished`. The instance was the only place its own journal and console
buffer existed.

So the controller now asks Girder whether a submission is still RUNNING on an instance
before it reaps it (`signals.running_jobs_by_instance`). If one is:

* the reap is logged at **WARNING**, naming the job and the age of its last heartbeat,
  and says plainly that the worker did *not* finish its work;
* `SIVACOR_DIAGNOSTICS_DIR` gets a dump — console buffer, Nova `fault`, and the action
  log — written **before** `delete_server`, because Nova drops the console with the
  server. `user_data` is stripped (it decodes to `MASTER_KEY_HEX` and `REDIS_PASSWORD`)
  and each file is `0600`, but treat the rest as secret-bearing: a console buffer is
  uncurated.

The instance is still deleted either way. It holds a quota slot, and the submission is
already unrecoverable — its workspace died with the VM. What changed is that the next
one is explainable.

In the dump, the action log is most useful for what is *missing*: a guest that ran
`systemctl poweroff` leaves no Nova action, so `create` alone means it shut itself down,
whereas a `stop` entry means something else did it. To watch a worker that is still
alive: `journalctl -u sivacor-worker-idle` on the box, or `openstack console log show
<id>` if you cannot get in — the unit logs to the serial console
(`deploy-sivacor@2d98cb7`) precisely because a keyless fleet once made this
undiagnosable.

## Not implemented yet

- **Prepull.** Workers are interchangeable on purpose: a submission's images are pulled
  inside the run, where the heartbeat covers the silence. Targeted prepull would mean
  the controller inspecting queue *contents* rather than depth, and instances ceasing
  to be interchangeable — a real trade, deferred deliberately.
- **Per-instance Redis ACL users**, which would make a leaked shared password inert and
  confine a compromised worker to its own queue.
