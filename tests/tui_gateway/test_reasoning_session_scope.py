"""Reasoning-effort session scoping in the TUI gateway (desktop backend).

Covers the "desktop reverts thinking to medium after one turn" report:

1. ``_session_info`` must report ``reasoning_effort: "none"`` when reasoning
   is disabled — reporting ``""`` (indistinguishable from "unset") made the
   desktop adopt the empty value after the first turn, wiping its sticky
   "thinking off" pick so every later chat reverted to the default effort.

2. ``config.set key=reasoning`` with a live session must be session-scoped:
   it must NOT rewrite the global ``agent.reasoning_effort`` in config.yaml
   (the desktop model menu applies a per-model preset on every selection,
   which was silently clobbering the user's configured value), and it must
   land on ``create_reasoning_override`` so lazily-built sessions (agent not
   constructed until the first prompt) don't drop the change.

3. ``_load_reasoning_config`` must honor a YAML boolean False
   (``reasoning_effort: false`` / ``off`` / ``no``) as thinking-disabled.
"""

from __future__ import annotations

import copy
import threading
from types import SimpleNamespace
from unittest.mock import patch

import tui_gateway.server as server
from tui_gateway.server import _session_info


def _agent(reasoning_config):
    return SimpleNamespace(
        reasoning_config=reasoning_config,
        service_tier=None,
        model="glm-5",
        provider="zai",
        session_id="sess-key",
    )


class TestSessionInfoReasoningEffort:
    """Disabled reasoning must be reported as 'none', never ''."""

    def test_disabled_reports_none(self) -> None:
        info = _session_info(_agent({"enabled": False}))
        assert info["reasoning_effort"] == "none"

    def test_enabled_reports_effort(self) -> None:
        info = _session_info(_agent({"enabled": True, "effort": "high"}))
        assert info["reasoning_effort"] == "high"

    def test_unset_reports_empty(self) -> None:
        info = _session_info(_agent(None))
        assert info["reasoning_effort"] == ""
        assert info["reasoning_effort_wire"] == ""

    def test_wire_level_is_what_the_route_actually_sends(self) -> None:
        """`ultra` is a Hermes-internal step (#61634): the route clamps it, and the Desktop must be able to
        say so ("ultra sends max on this route") instead of presenting Ultra as a distinct wire level."""
        info = _session_info(_agent({"enabled": True, "effort": "ultra"}))
        assert info["reasoning_effort"] == "ultra"
        assert info["reasoning_effort_wire"] == "max"
        # Verbatim levels report themselves, so clients only annotate a real clamp.
        assert _session_info(_agent({"enabled": True, "effort": "high"}))["reasoning_effort_wire"] == "high"
        assert _session_info(_agent({"enabled": False}))["reasoning_effort_wire"] == ""


class TestConfigSetReasoningSessionScope:
    """Session-targeted reasoning changes must not touch global config."""

    def _dispatch(self, params: dict) -> dict:
        handler = server._methods["config.set"]
        return handler("rid-1", params)

    def test_session_scoped_set_skips_global_write(self) -> None:
        agent = _agent(None)
        session = {"session_key": "k1", "agent": agent}
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"):
            resp = self._dispatch(
                {"key": "reasoning", "session_id": "s1", "value": "none"}
            )
        assert resp["result"]["value"] == "none"
        assert agent.reasoning_config == {"enabled": False}
        write_key.assert_not_called()


    def test_no_session_persists_globally(self) -> None:
        old_config = {"agent": {"reasoning_effort": "medium"}}
        writes = []
        with patch.object(server, "_load_cfg_raw", return_value=copy.deepcopy(old_config)), \
                patch.object(server, "_save_cfg", side_effect=lambda cfg: writes.append(copy.deepcopy(cfg))):
            resp = self._dispatch({"key": "reasoning", "value": "low"})
        assert resp["result"]["value"] == "low"
        assert writes[0]["agent"]["reasoning_effort"] == "low"

    def test_unknown_value_rejected(self) -> None:
        resp = self._dispatch({"key": "reasoning", "value": "bogus"})
        assert "error" in resp

    def test_inflight_build_finishes_before_reasoning_is_applied(self) -> None:
        ready = threading.Event()
        waiting = threading.Event()

        class ObservedReady:
            def wait(self, timeout=None):
                waiting.set()
                return ready.wait(timeout)

        session = {
            "session_key": "",
            "agent": None,
            "agent_build_started": True,
            "agent_ready": ObservedReady(),
            "agent_build_lock": threading.Lock(),
            "turn_route_binding": {
                "schema_version": "hermes.turn_route.binding.v1",
                "status": "routed",
                "owner": "middleware",
                "model": "gpt-6-luna",
                "provider": "openai-codex",
                "requested_provider": "openai-codex",
                "middleware_plugins": ["jev-router"],
                "middleware_reason": "economical/low",
                "reasoning_effort": "low",
                "reasoning_owner": "middleware",
            },
        }
        responses = []
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_emit"):
            worker = threading.Thread(target=lambda: responses.append(self._dispatch(
                {"key": "reasoning", "session_id": "s1", "value": "high"})))
            worker.start()
            assert waiting.wait(timeout=5)
            assert worker.is_alive()
            assert session["agent_build_lock"].acquire(timeout=1)
            session["agent_build_lock"].release()
            session["agent"] = _agent({"enabled": True, "effort": "medium"})
            ready.set()
            worker.join(timeout=5)

        assert not worker.is_alive()
        assert responses[0]["result"]["value"] == "high"
        assert session["agent"].reasoning_config == {"enabled": True, "effort": "high"}
        assert session["turn_route_binding"]["reasoning_effort"] == "high"
        assert session["turn_route_binding"]["reasoning_owner"] == "user"

    def test_global_inflight_timeout_never_exposes_tentative_config(self) -> None:
        writes = []
        session = {
            "session_key": "",
            "agent": None,
            "agent_build_started": True,
            "agent_build_lock": threading.Lock(),
        }

        class LateReady:
            def wait(self, timeout=None):
                assert writes == []
                session["agent"] = _agent({"enabled": True, "effort": "low"})
                return False

        session["agent_ready"] = LateReady()
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(
                    server, "_load_cfg_raw",
                    return_value={"agent": {"reasoning_effort": "low"}},
                ), \
                patch.object(server, "_save_cfg", side_effect=lambda cfg: writes.append(cfg)), \
                patch.object(server, "_persist_turn_route_state") as persist_state, \
                patch.object(server, "_emit"):
            resp = self._dispatch({
                "key": "reasoning", "session_id": "s1", "value": "high", "scope": "global"
            })

        assert resp["error"]["code"] == 5001
        assert writes == []
        persist_state.assert_not_called()
        assert session["agent"].reasoning_config == {"enabled": True, "effort": "low"}
        assert "create_reasoning_override" not in session
        assert "turn_route_binding" not in session

    def test_reasoning_and_binding_are_persisted_together(self) -> None:
        agent = _agent({"enabled": True, "effort": "low"})
        session = {
            "session_key": "k1",
            "agent": agent,
            "turn_route_binding": {
                "schema_version": "hermes.turn_route.binding.v1",
                "status": "routed",
                "owner": "middleware",
                "model": "gpt-6-luna",
                "provider": "openai-codex",
                "requested_provider": "openai-codex",
                "middleware_plugins": ["jev-router"],
                "middleware_reason": "economical/low",
                "reasoning_effort": "low",
                "reasoning_owner": "middleware",
            },
        }
        writes = []
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_persist_turn_route_state", side_effect=lambda state: writes.append(state) or True), \
                patch.object(server, "_emit"):
            resp = self._dispatch(
                {"key": "reasoning", "session_id": "s1", "value": "high"}
            )

        assert resp["result"]["value"] == "high"
        assert len(writes) == 1
        assert writes[0]["create_reasoning_override"] == {"enabled": True, "effort": "high"}
        assert writes[0]["turn_route_binding"]["reasoning_effort"] == "high"
        assert writes[0]["turn_route_binding"]["reasoning_owner"] == "user"
        assert writes[0]["turn_route_binding"]["middleware_plugins"] == ["jev-router"]

    def test_global_reasoning_rolls_back_config_when_session_write_fails(self) -> None:
        old_config = {"agent": {"reasoning_effort": "low"}}
        old_binding = {
            "schema_version": "hermes.turn_route.binding.v1",
            "status": "routed",
            "owner": "middleware",
            "model": "gpt-6-luna",
            "provider": "openai-codex",
            "requested_provider": "openai-codex",
            "middleware_plugins": ["jev-router"],
            "middleware_reason": "economical/low",
            "reasoning_effort": "low",
            "reasoning_owner": "middleware",
        }
        agent = _agent({"enabled": True, "effort": "low"})
        session = {
            "session_key": "k1",
            "agent": agent,
            "turn_route_binding": copy.deepcopy(old_binding),
        }
        store = {"config": copy.deepcopy(old_config), "writes": []}

        def save_cfg(cfg):
            store["writes"].append(copy.deepcopy(cfg))
            store["config"] = copy.deepcopy(cfg)

        def fail_after_unrelated_write(_state):
            server._write_config_key("display.theme", "dark")
            return False

        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_load_cfg_raw", side_effect=lambda: copy.deepcopy(store["config"])), \
                patch.object(server, "_save_cfg", side_effect=save_cfg), \
                patch.object(server, "_persist_turn_route_state", side_effect=fail_after_unrelated_write), \
                patch.object(server, "_emit"):
            resp = self._dispatch({
                "key": "reasoning", "session_id": "s1", "value": "high", "scope": "global"
            })

        assert resp["error"]["code"] == 5001
        assert store["config"]["agent"]["reasoning_effort"] == "low"
        assert store["config"]["display"]["theme"] == "dark"
        assert agent.reasoning_config == {"enabled": True, "effort": "low"}
        assert session["turn_route_binding"] == old_binding
        assert "create_reasoning_override" not in session

    def test_global_reasoning_write_failure_rolls_back_before_session_mutation(self) -> None:
        old_config = {"agent": {"reasoning_effort": "low"}}
        agent = _agent({"enabled": True, "effort": "low"})
        session = {"session_key": "k1", "agent": agent}
        store = {"config": copy.deepcopy(old_config), "calls": []}

        def fail_then_restore(cfg):
            store["calls"].append(copy.deepcopy(cfg))
            store["config"] = copy.deepcopy(cfg)
            if len(store["calls"]) == 1:
                raise OSError("config write failed")

        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_load_cfg_raw", side_effect=lambda: copy.deepcopy(store["config"])), \
                patch.object(server, "_save_cfg", side_effect=fail_then_restore), \
                patch.object(server, "_persist_turn_route_state") as persist_state, \
                patch.object(server, "_emit"):
            resp = self._dispatch({
                "key": "reasoning", "session_id": "s1", "value": "high", "scope": "global"
            })

        assert resp["error"]["code"] == 5001
        assert len(store["calls"]) == 2
        assert store["config"] == old_config
        persist_state.assert_not_called()
        assert agent.reasoning_config == {"enabled": True, "effort": "low"}
        assert "turn_route_binding" not in session
        assert "create_reasoning_override" not in session

    def test_global_reasoning_persists_same_effective_value_prebuild_and_live(self) -> None:
        old_config = {"agent": {"reasoning_effort": "low"}}
        for live in (False, True):
            binding = {
                "schema_version": "hermes.turn_route.binding.v1",
                "status": "routed",
                "owner": "middleware",
                "model": "gpt-6-luna",
                "provider": "openai-codex",
                "requested_provider": "openai-codex",
                "middleware_plugins": ["jev-router"],
                "middleware_reason": "economical/low",
                "reasoning_effort": "low",
                "reasoning_owner": "middleware",
            }
            session = {
                "session_key": "k1",
                "agent": _agent({"enabled": True, "effort": "low"}) if live else None,
                "turn_route_binding": binding,
            }
            writes = []
            store = {"config": copy.deepcopy(old_config)}

            def save_cfg(cfg):
                store["config"] = copy.deepcopy(cfg)

            with patch.dict(server._sessions, {"s1": session}, clear=False), \
                    patch.object(server, "_load_cfg_raw", side_effect=lambda: copy.deepcopy(store["config"])), \
                    patch.object(server, "_save_cfg", side_effect=save_cfg), \
                    patch.object(
                        server, "_persist_turn_route_state",
                        side_effect=lambda state: writes.append({
                            key: copy.deepcopy(value)
                            for key, value in state.items()
                            if key != "agent_build_lock"
                        }) or True,
                    ), \
                    patch.object(server, "_emit"):
                resp = self._dispatch({
                    "key": "reasoning", "session_id": "s1", "value": "high", "scope": "global"
                })

            assert resp["result"]["value"] == "high"
            assert writes[0]["create_reasoning_override"] == {"enabled": True, "effort": "high"}
            assert writes[0]["turn_route_binding"]["reasoning_effort"] == "high"
            assert writes[0]["turn_route_binding"]["reasoning_owner"] == "user"
            assert session["create_reasoning_override"] == {"enabled": True, "effort": "high"}
            if live:
                assert session["agent"].reasoning_config == {"enabled": True, "effort": "high"}
            else:
                # A later global edit cannot change this conversation before its
                # deferred first build. /new is the boundary that clears the pin.
                with patch.object(
                    server, "_load_reasoning_config",
                    return_value={"enabled": True, "effort": "low"},
                ):
                    build_kwargs = server._deferred_build_agent_kwargs(session, None)
                assert build_kwargs["reasoning_config_override"] == {
                    "enabled": True, "effort": "high"
                }

    def test_emit_failure_does_not_roll_back_committed_reasoning(self) -> None:
        old_config = {"agent": {"reasoning_effort": "low"}}
        store = {"config": copy.deepcopy(old_config)}
        agent = _agent({"enabled": True, "effort": "low"})
        session = {"session_key": "k1", "agent": agent}

        def save_cfg(cfg):
            store["config"] = copy.deepcopy(cfg)

        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_load_cfg_raw", side_effect=lambda: copy.deepcopy(store["config"])), \
                patch.object(server, "_save_cfg", side_effect=save_cfg), \
                patch.object(server, "_persist_turn_route_state", return_value=True), \
                patch.object(server, "_emit", side_effect=RuntimeError("notification failed")):
            resp = self._dispatch({
                "key": "reasoning", "session_id": "s1", "value": "high", "scope": "global"
            })

        assert resp["result"]["value"] == "high"
        assert store["config"]["agent"]["reasoning_effort"] == "high"
        assert agent.reasoning_config == {"enabled": True, "effort": "high"}
        assert session["create_reasoning_override"] == {"enabled": True, "effort": "high"}
        assert session["turn_route_binding"]["reasoning_effort"] == "high"


class TestLoadReasoningConfigYamlBoolean:
    """YAML `reasoning_effort: false` means disabled, not default."""

    def test_boolean_false_disables(self) -> None:
        with patch.object(
            server, "_load_cfg", return_value={"agent": {"reasoning_effort": False}}
        ):
            assert server._load_reasoning_config() == {"enabled": False}

    def test_string_false_disables(self) -> None:
        with patch.object(
            server, "_load_cfg", return_value={"agent": {"reasoning_effort": "false"}}
        ):
            assert server._load_reasoning_config() == {"enabled": False}
