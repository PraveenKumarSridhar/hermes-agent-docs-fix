"""``config.set model "X --reasoning <level>"`` on the TUI gateway: the effort rides with the pick.

Applied AFTER the live swap (``agent.switch_model`` re-resolves ``reasoning_config`` from
config.yaml) and scoped like the pick: a session pin by default, ``agent.reasoning_effort`` on
``--global``. The Ink TUI picker and the classic CLI both emit this exact shape.
"""

import copy
from types import SimpleNamespace

import pytest

import tui_gateway.server as server


class _Agent:
    def __init__(self):
        self.model, self.provider, self.base_url, self.api_key, self.api_mode = "old", "nous", "", "", ""
        self.reasoning_config = {"enabled": True, "effort": "medium"}

    def switch_model(self, **kw):
        self.model = kw["new_model"]
        self.provider = kw["new_provider"]
        self.reasoning_config = {"enabled": True, "effort": "medium"}  # re-resolved from config


@pytest.fixture
def _quiet_switch(monkeypatch):
    result = SimpleNamespace(
        success=True, new_model="new/model", target_provider="nous", base_url="", api_key="key",
        api_mode="chat_completions", warning_message="", model_info=None, error_message="",
        runtime_capabilities=None)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kw: result)
    monkeypatch.setattr("hermes_cli.model_cost_guard.expensive_model_warning", lambda *a, **k: None)
    for name in ("_restart_slash_worker", "_persist_live_session_runtime", "_persist_live_session_system_prompt",
                 "_append_model_switch_marker", "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    state = {
        "original_config": {
            "model": {"default": "old", "provider": "nous"},
            "agent": {"reasoning_effort": "medium"},
        },
        "config_writes": [],
    }
    state["current_config"] = copy.deepcopy(state["original_config"])

    def save_cfg(cfg):
        state["config_writes"].append(copy.deepcopy(cfg))
        state["current_config"] = copy.deepcopy(cfg)

    def write_key(key, value):
        current = state["current_config"]
        *parents, leaf = key.split(".")
        for parent in parents:
            current = current.setdefault(parent, {})
        current[leaf] = value

    monkeypatch.setattr(server, "_load_cfg_raw", lambda: copy.deepcopy(state["current_config"]))
    monkeypatch.setattr(server, "_save_cfg", save_cfg)
    monkeypatch.setattr(server, "_write_config_key", write_key)
    return state


def test_reasoning_flag_survives_the_swap_and_pins_the_session(_quiet_switch):
    agent = _Agent()
    session = {"agent": agent}

    out = server._apply_model_switch("sid", session, "new/model --provider nous --reasoning high --session")

    assert out["value"] == "new/model"
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}
    assert session["create_reasoning_override"] == {"enabled": True, "effort": "high"}
    assert session["turn_route_binding"]["reasoning_effort"] == "high"
    assert session["turn_route_binding"]["reasoning_owner"] == "user"
    assert _quiet_switch["config_writes"] == []


def test_reasoning_flag_with_global_writes_config_and_pins_current_conversation(_quiet_switch):
    agent = _Agent()
    session = {"agent": agent, "create_reasoning_override": {"enabled": True, "effort": "low"}}

    server._apply_model_switch("sid", session, "new/model --provider nous --reasoning none --global")

    assert agent.reasoning_config == {"enabled": False}
    assert session["turn_route_binding"]["reasoning_effort"] == "none"
    assert session["turn_route_binding"]["reasoning_owner"] == "user"
    assert _quiet_switch["config_writes"][0]["agent"]["reasoning_effort"] == "none"
    assert session["create_reasoning_override"] == {"enabled": False}


def test_global_config_write_failure_does_not_mutate_live_session(_quiet_switch, monkeypatch):
    agent = _Agent()
    session = {"agent": agent}
    calls = []

    def fail_then_restore(cfg):
        calls.append(copy.deepcopy(cfg))
        if len(calls) == 1:
            _quiet_switch["current_config"] = copy.deepcopy(cfg)
            raise OSError("config write failed")
        _quiet_switch["current_config"] = copy.deepcopy(cfg)

    monkeypatch.setattr(server, "_save_cfg", fail_then_restore)

    with pytest.raises(OSError, match="config write failed"):
        server._apply_model_switch(
            "sid", session, "new/model --provider nous --reasoning high --global")

    assert len(calls) == 2
    assert calls[1] == _quiet_switch["original_config"]
    assert _quiet_switch["current_config"] == _quiet_switch["original_config"]
    assert agent.model == "old"
    assert agent.provider == "nous"
    assert agent.reasoning_config == {"enabled": True, "effort": "medium"}
    assert "model_override" not in session
    assert "turn_route_binding" not in session


def test_global_session_failure_rolls_back_config_and_live_runtime(_quiet_switch, monkeypatch):
    agent = _Agent()
    old_binding = {
        "schema_version": "hermes.turn_route.binding.v1",
        "status": "routed",
        "owner": "middleware",
        "model": "old",
        "provider": "nous",
        "requested_provider": "nous",
        "middleware_plugins": ["jev-router"],
        "middleware_reason": "economical/medium",
        "reasoning_effort": "medium",
        "reasoning_owner": "middleware",
    }
    session = {"agent": agent, "turn_route_binding": copy.deepcopy(old_binding)}
    def fail_after_unrelated_write(_state):
        server._write_config_key("display.theme", "dark")
        return False

    monkeypatch.setattr(server, "_persist_turn_route_state", fail_after_unrelated_write)

    with pytest.raises(RuntimeError, match="binding could not be persisted"):
        server._apply_model_switch(
            "sid", session, "new/model --provider nous --reasoning high --global")

    assert len(_quiet_switch["config_writes"]) == 2
    assert _quiet_switch["current_config"]["model"] == _quiet_switch["original_config"]["model"]
    assert _quiet_switch["current_config"]["agent"] == _quiet_switch["original_config"]["agent"]
    assert _quiet_switch["current_config"]["display"]["theme"] == "dark"
    assert agent.model == "old"
    assert agent.provider == "nous"
    assert agent.reasoning_config == {"enabled": True, "effort": "medium"}
    assert session["turn_route_binding"] == old_binding
    assert "model_override" not in session


def test_post_commit_finalization_failure_keeps_committed_global_switch(_quiet_switch, monkeypatch):
    agent = _Agent()
    session = {"agent": agent}
    monkeypatch.setattr(
        server,
        "_finalize_agent_switch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("notification failed")),
    )
    monkeypatch.setattr(
        server,
        "_emit_session_info",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("notification failed")),
    )

    out = server._apply_model_switch(
        "sid", session, "new/model --provider nous --reasoning high --global")

    assert out["value"] == "new/model"
    assert _quiet_switch["current_config"]["model"]["default"] == "new/model"
    assert _quiet_switch["current_config"]["agent"]["reasoning_effort"] == "high"
    assert agent.model == "new/model"
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}
    assert session["model_override"]["model"] == "new/model"
    assert session["create_reasoning_override"] == {"enabled": True, "effort": "high"}
    assert session["turn_route_binding"]["model"] == "new/model"


def test_once_restore_marker_commits_before_finalization_failure(_quiet_switch, monkeypatch):
    agent = _Agent()
    session = {"agent": agent}
    monkeypatch.setattr(
        server,
        "_restart_slash_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("restart failed")),
    )

    out = server._apply_model_switch(
        "sid", session, "new/model --provider nous --reasoning high --once")

    assert out["scope"] == "once"
    assert session["one_turn_model_restore"]["model"] == "old"
    assert session["one_turn_model_restore"]["reasoning_config"] == {
        "enabled": True,
        "effort": "medium",
    }


def test_persistent_switch_clears_stale_once_marker_before_finalization_failure(
    _quiet_switch, monkeypatch
):
    agent = _Agent()
    session = {"agent": agent, "one_turn_model_restore": {"model": "stale"}}
    monkeypatch.setattr(
        server,
        "_restart_slash_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("restart failed")),
    )

    out = server._apply_model_switch(
        "sid", session, "new/model --provider nous --reasoning high --session")

    assert out["scope"] == "session"
    assert "one_turn_model_restore" not in session


def test_unknown_reasoning_flag_is_rejected(_quiet_switch):
    with pytest.raises(ValueError):
        server._apply_model_switch("sid", {"agent": _Agent()}, "new/model --reasoning turbo")
