"""
inference-api/app/ab_router.py

Fetches the active A/B config from the Experiment Tracker and decides
which model version to route each request to.

Routing logic:
  - Call GET /ab/active on the Experiment Tracker
  - If no active config (404) or tracker unreachable, fall back to the
    primary model (_model) with no routing
  - Otherwise, draw a random float in [0, 1):
      < split_weight  → route to model_a
      >= split_weight → route to model_b

The chosen model_id is logged with each request so traffic split can
be verified in logs and Prometheus.
"""

import logging
import os
import random
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

TRACKER_URL = os.environ.get(
    "EXPERIMENT_TRACKER_URL",
    "http://experiment-tracker.infergrid.svc.cluster.local:8001",
)
TRACKER_TIMEOUT = float(os.environ.get("EXPERIMENT_TRACKER_TIMEOUT", "0.5"))


@dataclass
class RoutingDecision:
    model_id: int | None      # None means no A/B config active
    model_version: str        # "model_a" | "model_b" | "primary"
    split_weight: float | None


def get_active_ab_config() -> dict | None:
    """
    Fetch the active A/B config from the Experiment Tracker.
    Returns the parsed JSON dict or None if unavailable.
    Times out after TRACKER_TIMEOUT seconds so inference latency is not impacted.
    """
    try:
        resp = requests.get(
            f"{TRACKER_URL}/ab/active",
            timeout=TRACKER_TIMEOUT,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        log.warning("Experiment Tracker unreachable, skipping A/B routing: %s", exc)
        return None


def make_routing_decision(config: dict | None) -> RoutingDecision:
    """
    Given an active A/B config (or None), decide which model to use.
    """
    if config is None:
        return RoutingDecision(
            model_id=None,
            model_version="primary",
            split_weight=None,
        )

    roll = random.random()
    split_weight = config["split_weight"]

    if roll < split_weight:
        return RoutingDecision(
            model_id=config["model_a_id"],
            model_version="model_a",
            split_weight=split_weight,
        )
    else:
        return RoutingDecision(
            model_id=config["model_b_id"],
            model_version="model_b",
            split_weight=split_weight,
        )