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

sivacor-autoscaler --dry-run     # decide and explain, change nothing
sivacor-autoscaler --once        # one iteration
sivacor-autoscaler               # loop
```

`--dry-run` runs the same `gather()` + `decide()` as a live round and skips only
execution, so it exercises the real code path rather than a parallel one. Start there.

## Not implemented yet

- **P3.3, the self-shutdown supervisor** that runs *on the worker* and powers it off
  when idle. Until it exists, instances are only reclaimed by the maximum-lifetime net,
  so watch `--dry-run` output and delete by hand.
- **Prepull.** Workers are interchangeable on purpose: a submission's images are pulled
  inside the run, where the heartbeat covers the silence. Targeted prepull would mean
  the controller inspecting queue *contents* rather than depth, and instances ceasing
  to be interchangeable — a real trade, deferred deliberately.
- **Per-instance Redis ACL users**, which would make a leaked shared password inert and
  confine a compromised worker to its own queue.
