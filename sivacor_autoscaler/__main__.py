"""Entry point. Configuration comes from the environment; secrets are never CLI args.

Run on the manager, where the broker, Girder and the OpenStack credentials all are.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

from .controller import Config, Controller
from .plan import Limits, decide


def _env(
    name: str, default: str | None = None, required: bool = False, hint: str = ""
) -> str | None:
    value = os.environ.get(name, default)
    if required and not value:
        sys.exit(f"{name} must be set{hint}")
    return value


def build_config() -> Config:
    return Config(
        template=Path(_env("SIVACOR_WORKER_TEMPLATE", required=True)),
        manager_ip=_env("SIVACOR_MANAGER_TENANT_IP", required=True),
        master_key_hex=_env("MASTER_KEY_HEX", required=True),
        redis_password=_env("REDIS_PASSWORD", required=True),
        # Required, deliberately. Defaulting it would put two deployments in one tag
        # namespace, and the failure that produces -- each controller counting and
        # reaping the other's workers -- is silent. See fleet.DEPLOYMENT_TAG_PREFIX.
        # The stack passes the deployment's `domain`.
        #
        # Being required also makes this image and the stack file a matched pair: a
        # deployment that takes the new image with an old `docker-stack.autoscaler.yml`
        # gets a container that exits here and, under `restart_policy: any`, crash-loops
        # -- which means NO controller, so nothing scales and nothing is reaped. Loud in
        # `docker service logs wt_autoscaler` and invisible everywhere else, so the
        # message has to name the fix. Same shape as P0.3's plugin/stack pairing.
        deployment=_env(
            "SIVACOR_DEPLOYMENT",
            required=True,
            hint=(
                ", normally the deployment domain (e.g. sivacor.org). It scopes which "
                "OpenStack instances this controller owns; deploy-sivacor's Makefile "
                "derives it from `domain`, so if you are seeing this, the stack file is "
                "older than this image -- update deploy-sivacor and redeploy together."
            ),
        ),
        dispatch_queue=_env("SIVACOR_DISPATCH_QUEUE", "sivacor"),
        girder_host=_env("SIVACOR_GIRDER_HOST"),
        worker_image=_env("SIVACOR_WORKER_IMAGE"),
        worker_queues=_env("SIVACOR_WORKER_QUEUES"),
        image=_env("SIVACOR_OS_IMAGE", "Featured-Ubuntu24"),
        flavor=_env("SIVACOR_OS_FLAVOR", "m3.medium"),
        network=_env("SIVACOR_OS_NETWORK", "auto_allocated_network"),
        key_name=_env("SIVACOR_OS_KEYPAIR"),
        security_groups=[
            g for g in (_env("SIVACOR_OS_SECGROUPS", "") or "").split(",") if g
        ],
        interval=float(_env("SIVACOR_INTERVAL", "30")),
        # Unset = no capture, which is the pre-2026-08-11 behaviour rather than a
        # neutral default: without it an instance that powered off mid-run is deleted
        # with its console buffer, and nothing anywhere records why it died. Set it to
        # a bind-mounted path -- a directory inside the container dies with it.
        diagnostics_dir=(
            Path(d) if (d := _env("SIVACOR_DIAGNOSTICS_DIR")) else None
        ),
        limits=Limits(
            max_instances=int(_env("SIVACOR_MAX_INSTANCES", "5")),
            max_lifetime=timedelta(hours=float(_env("SIVACOR_MAX_LIFETIME_HOURS", "30"))),
            breaker_threshold=int(_env("SIVACOR_BREAKER_THRESHOLD", "3")),
            # Unset = the D9 check is off, which is the safe default: arming it
            # against a worker image that does not write readiness markers deletes
            # healthy instances. See Limits.provision_deadline.
            provision_deadline=(
                timedelta(minutes=float(mins))
                if (mins := _env("SIVACOR_PROVISION_DEADLINE_MINUTES"))
                else None
            ),
            # S6's other two quota dimensions. Unset = that check is off, which is the
            # pre-P3 behaviour and correct for a uniform fleet (D3): the instance count
            # binds at the bottom rung and only there.
            #
            # Set them **below** the real OpenStack quota, for the same reason
            # max_instances is: the same 320 vCPU / 1220 GiB carry the manager, the test
            # mirror and any hand-made debug VM, so deriving these from the quota
            # guarantees a collision with ordinary work. Live quota 2026-08-20:
            # 25 instances / 320 vCPU / 1220 GiB, with 3 / 12 / 45 GiB already in use by
            # the two managers.
            max_vcpus=(int(v) if (v := _env("SIVACOR_MAX_VCPUS")) else None),
            max_ram_gb=(int(v) if (v := _env("SIVACOR_MAX_RAM_GB")) else None),
        ),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--once",
        action="store_true",
        help="run a single iteration and exit (for cron or a smoke test)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report the decision without creating or deleting anything",
    )
    p.add_argument("--cloud", help="clouds.yaml entry; omit to use OS_* env vars")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="debug logging for this package only; HTTP client logs stay at INFO",
    )
    args = p.parse_args()

    # -v raises ONLY this package. It used to set the root logger to DEBUG, which
    # made keystoneauth log every HTTP request body -- including the Nova POST that
    # carries the worker user-data. That is base64, not encryption: one decode yields
    # MASTER_KEY_HEX and REDIS_PASSWORD in cleartext, and it reached a shared log on
    # 2026-08-01. Secrets travel in user-data (P2.2), so any library that logs a
    # request body is a secret sink. -v means "explain your decisions", never "dump
    # every HTTP body".
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if args.verbose:
        logging.getLogger(__package__).setLevel(logging.DEBUG)
    # Belt and braces: pin the request-body loggers even if the root level is raised
    # by a future edit or an outer harness.
    for noisy in ("keystoneauth", "openstack", "urllib3", "requests", "swiftclient"):
        logging.getLogger(noisy).setLevel(logging.INFO)

    cfg = build_config()
    if not cfg.template.is_file():
        sys.exit(f"worker template not found: {cfg.template}")

    # Worth a line at INFO: this is the value that decides which instances the process
    # will create, count and *delete*, and the one mistake it protects against -- two
    # deployments sharing a project -- is invisible from inside either one.
    from . import fleet as _fleet

    logging.getLogger(__name__).info(
        "owning instances tagged %r; ignoring every other %r instance",
        _fleet.deployment_tag(cfg.deployment),
        _fleet.FLEET_TAG,
    )

    # Not fatal -- a keyless fleet is a legitimate production posture, and workers
    # hold no key material worth reaching for. But it is a terrible *debugging*
    # posture, and fleet.create_instance omits key_name silently when it is unset,
    # so the first time anyone finds out is when a worker misbehaves and there is no
    # way in. That happened on 2026-08-01. Warn here rather than at create time: it
    # fires once, before anything is spent, and covers --dry-run too.
    if cfg.key_name:
        logging.getLogger(__name__).info("workers get keypair %r", cfg.key_name)
    else:
        logging.getLogger(__name__).warning(
            "SIVACOR_OS_KEYPAIR is unset: workers launch with NO ssh key. A worker "
            "that fails to power itself off is then diagnosable only from "
            "`openstack console log show` and the broker."
        )

    # Third of the same family: a capability that is off by default has to announce
    # itself, or the first time anyone learns it was off is while trying to read the
    # dump that was never written.
    if cfg.diagnostics_dir:
        logging.getLogger(__name__).info(
            "pre-delete diagnostics go to %s; keep it bind-mounted and treat it as "
            "secret-bearing (console buffers are uncurated)",
            cfg.diagnostics_dir,
        )
    else:
        logging.getLogger(__name__).warning(
            "SIVACOR_DIAGNOSTICS_DIR is unset: an instance that powers off mid-run is "
            "deleted along with the only copy of its console log, so 'reaped for no "
            "heartbeat' will stay unexplainable. Observed three times on 2026-08-10/11."
        )

    # Same reasoning as the keypair warning: state it once, up front. Off is the safe
    # default but it is also the state in which run 6's phantom recurs, so a silent
    # default here would be the second time this project shipped an inert signal.
    if cfg.limits.provision_deadline is not None:
        logging.getLogger(__name__).info(
            "unprovisioned instances are reaped after %s; requires a worker image "
            "that announces readiness (plan D9)",
            cfg.limits.provision_deadline,
        )
    else:
        logging.getLogger(__name__).warning(
            "SIVACOR_PROVISION_DEADLINE_MINUTES is unset: an instance that boots but "
            "fails to provision will be counted as available capacity until the %s "
            "max-lifetime sweep, stalling submissions behind it (plan D9).",
            cfg.limits.max_lifetime,
        )

    # Fifth of the same family, added because its absence was noticed the hard way:
    # `SIVACOR_MAX_VCPUS=8` was confirmed present only by `docker exec ... env`, because
    # nothing announced it and the quota stop is by design a line that fires rarely. A
    # guardrail whose configured value never reaches the log is one nobody can confirm
    # took effect until it bites -- and for this one "it bit" looks like a submission
    # waiting, which is the state S7 says must never be silent.
    if cfg.limits.max_vcpus or cfg.limits.max_ram_gb:
        logging.getLogger(__name__).info(
            "quota headroom: max_instances=%s, max_vcpus=%s, max_ram_gb=%s (S6). "
            "Whichever binds first stops creation, oldest submission first",
            cfg.limits.max_instances,
            cfg.limits.max_vcpus if cfg.limits.max_vcpus else "off",
            cfg.limits.max_ram_gb if cfg.limits.max_ram_gb else "off",
        )
    else:
        logging.getLogger(__name__).info(
            "quota headroom is the instance count alone (max_instances=%s); "
            "SIVACOR_MAX_VCPUS and SIVACOR_MAX_RAM_GB are unset. Correct while every "
            "worker is one shape -- the count binds at the bottom rung and only there "
            "-- but it stops bounding cost once sizes differ: at 125 GiB the RAM quota "
            "binds at nine instances, not %s (S6, superseding D3)",
            cfg.limits.max_instances,
            cfg.limits.max_instances,
        )

    # Sixth of the same family, and the one that fails latest if left unsaid. Once
    # targeted assignment is armed -- which is a Girder setting, not an environment
    # variable, so it can happen without a redeploy and without this process
    # restarting -- the controller publishes celery chains itself. Without a broker it
    # claims the submission and then cannot send it, so the submission is bound to a
    # worker that is never told and waits for the server-side reaper. Warn now, while
    # nothing is at stake, rather than at the first assignment.
    if not os.environ.get("GIRDER_WORKER_BROKER"):
        logging.getLogger(__name__).warning(
            "GIRDER_WORKER_BROKER is unset: this controller can decide assignments "
            "but cannot publish them. Set it (and GIRDER_WORKER_BACKEND) to the same "
            "broker girder and local_worker use before arming "
            "sivacor.targeted_assignment."
        )

    import openstack
    import redis as redis_lib
    from pymongo import MongoClient

    conn = openstack.connect(cloud=args.cloud) if args.cloud else openstack.connect()
    # How *we* reach the broker is not how *workers* reach it, and conflating the two
    # is what would force production to publish 6379. A worker is off-box and must use
    # the manager's tenant address; this process may be running inside the stack, where
    # the `redis` service name resolves on the overlay network and no published port is
    # needed at all. `SIVACOR_MANAGER_TENANT_IP` therefore keeps its one real job --
    # the value injected into worker user-data -- and this is a separate knob.
    #
    # Default preserves the pre-container behaviour exactly, so running from a checkout
    # on the manager needs no new configuration.
    redis_url = _env(
        "SIVACOR_REDIS_URL", f"redis://:{cfg.redis_password}@{cfg.manager_ip}:6379/"
    )
    redis_client = redis_lib.Redis.from_url(redis_url)
    # Girder's database, not its REST API -- no API key to bootstrap and no endpoint
    # scoping to get wrong. See signals.JOB_COLLECTION. `tz_aware` matters: Girder
    # writes tz-aware datetimes and pymongo hands back naive ones without it, which
    # turns every timestamp comparison into a silent TypeError. The server-side reaper
    # hit exactly that and has to normalise by hand.
    mongo = MongoClient(_env("GIRDER_MONGO_URI", "mongodb://mongo:27017/girder"),
                        tz_aware=True)
    db = mongo.get_default_database()

    controller = Controller(conn, redis_client, db, cfg)

    # Before anything is created, and fatal on failure. The catalogue names flavours
    # this process will hand to Nova, and an unknown one raises inside create_instance,
    # which counts towards the circuit breaker -- three of those stop the entire fleet
    # (P1). Failing here instead makes a typo a boot error on one deployment, which is
    # both louder and cheaper. Covers --dry-run too, deliberately: a dry run whose
    # catalogue would not boot is not a useful dry run.
    #
    # `RuntimeError` only, which is every failure `catalogue` raises on purpose: an
    # empty or malformed setting, and a rung that disagrees with Nova. Anything else --
    # girder not importable, keystone refusing the credential -- keeps its traceback,
    # because for those the cause matters more than the message and a one-line exit
    # would hide it.
    try:
        controller.load_catalogue()
    except RuntimeError as exc:
        sys.exit(f"worker size catalogue unusable: {exc}")

    if args.dry_run:
        # Deliberately reuses gather() + decide() so a dry run exercises the same code
        # path as a live one; only execution is skipped.
        state = controller.gather()
        decision = decide(state, cfg.limits)
        print(f"depth={state.queue_depth} serving={state.serving} live/total="
              f"{sum(i.is_live for i in state.instances)}/{len(state.instances)}")
        for reason in decision.reasons:
            print(f"  {reason}")
        shapes = ", ".join(
            f"{r} GB" if r is not None else "unsized" for r in decision.create
        )
        print(
            f"would create {len(decision.create)}"
            + (f" ({shapes})" if decision.create else "")
            + f", delete {len(decision.delete)}"
        )
        return 0

    if args.once:
        controller.step()
        return 0
    controller.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
