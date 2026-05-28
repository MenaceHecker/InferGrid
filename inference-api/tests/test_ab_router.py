import pytest
import requests

from app.ab_router import RoutingDecision, get_active_ab_config, make_routing_decision

# Sample A/B config fixture

SAMPLE_CONFIG = {
    "id": 1,
    "model_a_id": 10,
    "model_b_id": 20,
    "split_weight": 0.7,
    "active": True,
}


# get_active_ab_config


def test_get_active_ab_config_returns_config_on_200(monkeypatch):
    class MockResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return SAMPLE_CONFIG

    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
    config = get_active_ab_config()
    assert config == SAMPLE_CONFIG


def test_get_active_ab_config_returns_none_on_404(monkeypatch):
    class MockResponse:
        status_code = 404

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
    config = get_active_ab_config()
    assert config is None


def test_get_active_ab_config_returns_none_on_connection_error(monkeypatch):
    def raise_error(*a, **kw):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", raise_error)
    config = get_active_ab_config()
    assert config is None


def test_get_active_ab_config_returns_none_on_timeout(monkeypatch):
    def raise_timeout(*a, **kw):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(requests, "get", raise_timeout)
    config = get_active_ab_config()
    assert config is None


# make_routing_decision — no config


def test_routing_no_config_returns_primary():
    decision = make_routing_decision(None)
    assert decision.model_version == "primary"
    assert decision.model_id is None
    assert decision.split_weight is None

# make_routing_decision — with config


def test_routing_routes_to_model_a_when_roll_below_split(monkeypatch):
    monkeypatch.setattr("app.ab_router.random.random", lambda: 0.1)
    decision = make_routing_decision(SAMPLE_CONFIG)
    assert decision.model_version == "model_a"
    assert decision.model_id == SAMPLE_CONFIG["model_a_id"]


def test_routing_routes_to_model_b_when_roll_above_split(monkeypatch):
    monkeypatch.setattr("app.ab_router.random.random", lambda: 0.9)
    decision = make_routing_decision(SAMPLE_CONFIG)
    assert decision.model_version == "model_b"
    assert decision.model_id == SAMPLE_CONFIG["model_b_id"]


def test_routing_boundary_at_split_weight_goes_to_model_b(monkeypatch):
    # roll == split_weight → model_b (not strictly less than)
    monkeypatch.setattr("app.ab_router.random.random", lambda: 0.7)
    decision = make_routing_decision(SAMPLE_CONFIG)
    assert decision.model_version == "model_b"


def test_routing_returns_routing_decision_type():
    decision = make_routing_decision(SAMPLE_CONFIG)
    assert isinstance(decision, RoutingDecision)


def test_routing_carries_split_weight():
    decision = make_routing_decision(SAMPLE_CONFIG)
    assert decision.split_weight == pytest.approx(0.7)


def test_routing_split_approximates_configured_ratio(monkeypatch):
    """Over 10,000 draws, routing ratio should be within 2% of split_weight."""
    import random as stdlib_random

    monkeypatch.setattr("app.ab_router.random.random", stdlib_random.random)
    stdlib_random.seed(42)

    config = {**SAMPLE_CONFIG, "split_weight": 0.6}
    model_a_count = sum(
        1
        for _ in range(10_000)
        if make_routing_decision(config).model_version == "model_a"
    )
    ratio = model_a_count / 10_000
    assert abs(ratio - 0.6) < 0.02