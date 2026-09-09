"""DEMO_MODE opens the deployment so a link can be handed to an audience.

It does two things that have to stay in step: it stops enforcing the API key on every agent
endpoint, and it tells the browser to hide its connection settings — because a page with no key
to enter should not show a field demanding one, and a page that hides the field must not then be
rejected for having no key.

The direction of every default here is deliberate. Off unless explicitly switched on, and any
ambiguity resolves to CLOSED: an accidentally open deployment spends this project's Earth Engine
quota and LLM budget for whoever finds the URL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import api.server as server  # noqa: E402

KEY = "s3cret-key"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("AGENT_CHAT_API_KEY", raising=False)


def _check(headers: dict | None = None):
    """Run the guard the endpoints run, inside a request carrying these headers."""
    with server.app.test_request_context("/agent/chat", headers=headers or {}):
        server._require_agent_chat_api_key()


def _ui_config() -> dict:
    with server.app.test_client() as client:
        res = client.get("/agent/ui-config")
        assert res.status_code == 200
        return res.get_json()


# --- the switch itself ----------------------------------------------------------
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " true "])
def test_the_spellings_that_turn_it_on(monkeypatch, value):
    monkeypatch.setenv("DEMO_MODE", value)
    assert server._demo_mode() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe", "demo"])
def test_everything_else_leaves_it_off(monkeypatch, value):
    """Anything unrecognised reads as OFF. A typo must not open the deployment."""
    monkeypatch.setenv("DEMO_MODE", value)
    assert server._demo_mode() is False


def test_off_when_the_variable_is_absent():
    assert server._demo_mode() is False


# --- what it does to the key ----------------------------------------------------
def test_a_configured_key_is_enforced_when_demo_mode_is_off(monkeypatch):
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    with pytest.raises(PermissionError):
        _check()
    with pytest.raises(PermissionError):
        _check({"X-API-KEY": "wrong"})
    _check({"X-API-KEY": KEY})           # the real key still works
    _check({"Authorization": f"Bearer {KEY}"})


def test_demo_mode_lets_a_request_through_with_no_key(monkeypatch):
    """The key stays CONFIGURED. A demo should not require unsetting the secret and

    remembering to put it back — that is how a deployment ends up permanently open.
    """
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    monkeypatch.setenv("DEMO_MODE", "true")
    _check()
    _check({"X-API-KEY": "wrong"})       # not merely optional: not checked at all


def test_no_key_configured_is_open_with_or_without_demo_mode(monkeypatch):
    """Unchanged behaviour: an unset key has always meant no auth."""
    _check()
    monkeypatch.setenv("DEMO_MODE", "true")
    _check()


# --- what it tells the browser --------------------------------------------------
def test_ui_config_says_a_key_is_required_when_one_is_enforced(monkeypatch):
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    assert _ui_config() == {"demo_mode": False, "api_key_required": True}


def test_ui_config_stops_asking_for_a_key_in_demo_mode(monkeypatch):
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    monkeypatch.setenv("DEMO_MODE", "true")
    assert _ui_config() == {"demo_mode": True, "api_key_required": False}


def test_ui_config_needs_no_key_of_its_own(monkeypatch):
    """Unauthenticated on purpose: a client that cannot authenticate is exactly the one that

    needs to ask whether it has to.
    """
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    with server.app.test_client() as client:
        assert client.get("/agent/ui-config").status_code == 200


def test_ui_config_never_returns_the_key_itself(monkeypatch):
    monkeypatch.setenv("AGENT_CHAT_API_KEY", KEY)
    monkeypatch.setenv("DEMO_MODE", "true")
    assert KEY not in str(_ui_config())
