"""The control loop: gather signals, ask :mod:`plan`, execute, repeat.

Deliberately dull. Every judgement lives in :func:`plan.decide`; this only does I/O
and error classification.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import fleet, signals
from .plan import FleetState, Limits, decide

logger = logging.getLogger(__name__)


@dataclass
class Config:
    template: Path
    manager_ip: str
    master_key_hex: str
    redis_password: str
    dispatch_queue: str = "sivacor"
    girder_host: str | None = None
    worker_image: str | None = None
    image: str = "Featured-Ubuntu24"
    flavor: str = "m3.medium"
    network: str = "auto_allocated_network"
    key_name: str | None = None
    security_groups: list[str] = field(default_factory=list)
    interval: float = 30.0
    limits: Limits = field(default_factory=Limits)


class Controller:
    def __init__(self, conn, redis_client, girder_client, cfg: Config):
        self.conn = conn
        self.redis = redis_client
        self.girder = girder_client
        self.cfg = cfg
        #: Consecutive instances that booted but never registered with celery. Reset
        #: by any successful round, so a transient failure does not accumulate towards
        #: the breaker.
        self.consecutive_failures = 0

    def gather(self) -> FleetState:
        return FleetState(
            queue_depth=signals.queue_depth(self.redis, self.cfg.dispatch_queue),
            serving=signals.serving_count(self.girder),
            spent=signals.spent_instance_ids(self.girder, self.cfg.dispatch_queue),
            instances=fleet.list_fleet(self.conn),
            consecutive_failures=self.consecutive_failures,
        )

    def step(self) -> None:
        """One iteration. Never raises for an expected condition."""
        try:
            state = self.gather()
        except Exception:
            # Deciding on partial information is how a controller creates instances it
            # does not need. Skip the round; the next one is 30 s away.
            logger.warning("skipping round: could not gather state", exc_info=True)
            return

        decision = decide(state, self.cfg.limits)
        for reason in decision.reasons:
            logger.info("%s", reason)

        # Deletes first: they free slots the creates may want, and they must happen
        # even when the breaker has blocked creation.
        for instance_id in decision.delete:
            try:
                fleet.delete_instance(self.conn, instance_id)
            except Exception:
                logger.warning("could not delete %s", instance_id, exc_info=True)

        for _ in range(decision.create):
            try:
                user_data = fleet.build_user_data(
                    self.cfg.template,
                    master_key_hex=self.cfg.master_key_hex,
                    redis_password=self.cfg.redis_password,
                    manager_ip=self.cfg.manager_ip,
                    girder_host=self.cfg.girder_host,
                    worker_image=self.cfg.worker_image,
                )
                fleet.create_instance(self.conn, self.cfg, user_data)
            except fleet.QuotaExceeded as exc:
                # Backpressure, not failure: the submission stays queued and this must
                # not count towards the breaker, or a full allocation would stop the
                # fleet scaling precisely when it is busiest.
                logger.info("allocation full, leaving work queued: %s", exc)
                break
            except Exception:
                self.consecutive_failures += 1
                logger.warning(
                    "instance creation failed (%d consecutive)",
                    self.consecutive_failures,
                    exc_info=True,
                )
                break

    def run(self) -> None:
        logger.info(
            "controller starting: queue=%s cap=%d interval=%.0fs",
            self.cfg.dispatch_queue,
            self.cfg.limits.max_instances,
            self.cfg.interval,
        )
        while True:
            self.step()
            time.sleep(self.cfg.interval)
