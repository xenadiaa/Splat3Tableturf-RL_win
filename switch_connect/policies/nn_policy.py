from __future__ import annotations

import importlib
import json
import shlex
import subprocess
from dataclasses import asdict
from typing import Callable, Dict, Tuple

from src.engine.env_core import Action
from vision_capture.state_types import ObservedState


def _parse_module_ref(ref: str) -> Tuple[str, str]:
    if ":" not in ref:
        raise ValueError("module ref must be in format 'module.path:function_name'")
    mod, fn = ref.split(":", 1)
    if not mod or not fn:
        raise ValueError("invalid module ref")
    return mod, fn


def _action_from_dict(payload: Dict[str, object]) -> Action:
    return Action(
        player=str(payload.get("player", "P1")),
        card_number=int(payload["card_number"]) if payload.get("card_number") is not None else None,
        surrender=bool(payload.get("surrender", False)),
        pass_turn=bool(payload.get("pass_turn", False)),
        use_sp_attack=bool(payload.get("use_sp_attack", False)),
        rotation=int(payload.get("rotation", 0)),
        x=int(payload["x"]) if payload.get("x") is not None else None,
        y=int(payload["y"]) if payload.get("y") is not None else None,
    )


def choose_action_nn_module(obs: ObservedState, module_ref: str):
    """
    Load a python callable and ask it for action.
    callable signature:
      def policy(observed_state_dict: dict) -> dict
    """
    module_name, fn_name = _parse_module_ref(module_ref)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"callable not found: {module_ref}")
    action_dict = fn(asdict(obs))
    if not isinstance(action_dict, dict):
        raise ValueError("NN module policy must return dict")
    return _action_from_dict(action_dict)


def choose_action_nn_command(obs: ObservedState, command: str):
    """
    Run an external process for NN inference.
    Process contract:
    - stdin: ObservedState JSON
    - stdout: Action JSON object
    """
    argv = shlex.split(command)
    if not argv:
        raise ValueError("empty nn command")
    proc = subprocess.run(
        argv,
        input=json.dumps(asdict(obs), ensure_ascii=False),
        text=True,
        capture_output=True,
        check=True,
    )
    out = proc.stdout.strip()
    if not out:
        raise ValueError("nn command produced empty stdout")
    action_dict = json.loads(out)
    if not isinstance(action_dict, dict):
        raise ValueError("nn command must output action json object")
    return _action_from_dict(action_dict)
