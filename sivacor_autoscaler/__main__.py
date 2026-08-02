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


def _env(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.environ.get(name, default)
    if required and not value:
        sys.exit(f"{name} must be set")
    return value


def build_config() -> Config:
    return Config(
        template=Path(_env("SIVACOR_WORKER_TEMPLATE", required=True)),
        manager_ip=_env("SIVACOR_MANAGER_TENANT_IP", required=True),
        master_key_hex=_env("MASTER_KEY_HEX", required=True),
        redis_password=_env("REDIS_PASSWORD", required=True),
        dispatch_queue=_env("SIVACOR_DISPATCH_QUEUE", "sivacor"),
        girder_host=_env("SIVACOR_GIRDER_HOST"),
        worker_image=_env("SIVACOR_WORKER_IMAGE"),
        image=_env("SIVACOR_OS_IMAGE", "Featured-Ubuntu24"),
        flavor=_env("SIVACOR_OS_FLAVOR", "m3.medium"),
        network=_env("SIVACOR_OS_NETWORK", "auto_allocated_network"),
        key_name=_env("SIVACOR_OS_KEYPAIR"),
        security_groups=[
            g for g in (_env("SIVACOR_OS_SECGROUPS", "") or "").split(",") if g
        ],
        interval=float(_env("SIVACOR_INTERVAL", "30")),
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

    import openstack
    import redis as redis_lib
    from girder_client import GirderClient

    conn = openstack.connect(cloud=args.cloud) if args.cloud else openstack.connect()
    redis_client = redis_lib.Redis.from_url(
        f"redis://:{cfg.redis_password}@{cfg.manager_ip}:6379/"
    )
    girder = GirderClient(apiUrl=_env("GIRDER_API_URL", required=True))
    if api_key := _env("GIRDER_API_KEY"):
        girder.authenticate(apiKey=api_key)

    controller = Controller(conn, redis_client, girder, cfg)

    if args.dry_run:
        # Deliberately reuses gather() + decide() so a dry run exercises the same code
        # path as a live one; only execution is skipped.
        state = controller.gather()
        decision = decide(state, cfg.limits)
        print(f"depth={state.queue_depth} serving={state.serving} live/total="
              f"{sum(i.is_live for i in state.instances)}/{len(state.instances)}")
        for reason in decision.reasons:
            print(f"  {reason}")
        print(f"would create {decision.create}, delete {len(decision.delete)}")
        return 0

    if args.once:
        controller.step()
        return 0
    controller.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
