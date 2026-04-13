"""Strategy registry for NPC defaults, NN checkpoints, and auto players."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from src.engine.env_core import Action, choose_default_strategy_action, legal_actions

STRATEGY_DIR = Path(__file__).resolve().parent
NPC_DEFAULTS_JSON = STRATEGY_DIR / "npc_defaults.json"
DEFAULT_DEFS_JSON = STRATEGY_DIR / "default_strategy_defs.json"


def _load_module_strategy_meta(module_name: str) -> Dict[str, str]:
    try:
        module = importlib.import_module(f"src.strategy.{module_name}")
    except Exception:
        return {}
    label = getattr(module, "STRATEGY_LABEL", None)
    strategy_id = getattr(module, "STRATEGY_ID", None)
    out: Dict[str, str] = {}
    if isinstance(label, str) and label.strip():
        out["label"] = label.strip()
    if isinstance(strategy_id, str) and strategy_id.strip():
        out["id"] = strategy_id.strip()
    return out


def load_npc_strategy_table() -> List[dict]:
    data = json.loads(NPC_DEFAULTS_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("npc_defaults.json root must be list")
    return data


def _load_default_defs() -> List[dict]:
    data = json.loads(DEFAULT_DEFS_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("default_strategy_defs.json root must be list")
    return data


def resolve_npc_nn_spec(npc_name: str) -> Optional[Dict[str, object]]:
    pt_path = STRATEGY_DIR / f"{npc_name}_nn.pt"
    if not pt_path.exists():
        return None
    loader_path = STRATEGY_DIR / "nn_loader.py"
    if not loader_path.exists():
        return None
    return {
        "type": "python_module",
        "module_file": str(loader_path),
        "function": "choose_action",
        "checkpoint_file": str(pt_path),
        "npc_name": npc_name,
    }


def list_available_strategy_ids() -> List[dict]:
    rows: List[dict] = []
    rows.extend(_load_default_defs())
    for path in sorted(STRATEGY_DIR.glob("*_nn.pt")):
        npc_name = path.stem[:-3] if path.stem.endswith("_nn") else path.stem
        rows.append(
            {
                "id": f"nn:{npc_name}",
                "kind": "nn",
                "label": f"NN {npc_name}",
                "npc_name": npc_name,
                "path": str(path),
            }
        )
    for path in sorted(STRATEGY_DIR.glob("*_strategy.py")):
        mod_name = path.stem
        meta = _load_module_strategy_meta(mod_name)
        rows.append(
            {
                "id": meta.get("id", f"module:{mod_name}"),
                "kind": "module",
                "label": meta.get("label", f"Module {mod_name}"),
                "module_name": mod_name,
                "path": str(path),
            }
        )
    return rows


def _strategy_row_by_id(strategy_id: str) -> Optional[dict]:
    for row in list_available_strategy_ids():
        if row.get("id") == strategy_id:
            return row
    return None


def choose_action_from_strategy_id(state, player: str, strategy_id: str) -> Action:
    row = _strategy_row_by_id(strategy_id)
    if row is None:
        raise ValueError(f"unknown strategy id: {strategy_id}")

    if row["kind"] == "default":
        return choose_default_strategy_action(
            state=state,
            player=player,
            style=str(row["style"]),
            level=str(row["level"]),
        )

    actions = legal_actions(state, player)
    if not actions:
        ps = state.players[player]
        return Action(player=player, card_number=ps.hand[0].Number, pass_turn=True)

    if row["kind"] == "nn":
        from . import nn_loader, strategic_nn_loader

        legal_payload = [{
            "player": a.player,
            "card_number": a.card_number,
            "surrender": a.surrender,
            "pass_turn": a.pass_turn,
            "use_sp_attack": a.use_sp_attack,
            "rotation": a.rotation,
            "x": a.x,
            "y": a.y,
        } for a in actions]
        checkpoint_file = str(row["path"])
        if strategic_nn_loader.is_strategic_checkpoint(checkpoint_file):
            payload = strategic_nn_loader.choose_action(
                state=state,
                player=player,
                legal_actions=legal_payload,
                context={"checkpoint_file": checkpoint_file},
            )
        else:
            payload = nn_loader.choose_action(
                state=state,
                player=player,
                legal_actions=legal_payload,
                context={"checkpoint_file": checkpoint_file},
            )
        for a in actions:
            if (
                a.card_number == payload.get("card_number")
                and a.pass_turn == bool(payload.get("pass_turn", False))
                and a.use_sp_attack == bool(payload.get("use_sp_attack", False))
                and a.rotation == int(payload.get("rotation", 0))
                and a.x == payload.get("x")
                and a.y == payload.get("y")
            ):
                return a
        raise RuntimeError(f"strategy returned non-legal action: {strategy_id}")

    if row["kind"] == "module":
        module = importlib.import_module(f"src.strategy.{row['module_name']}")
        fn = getattr(module, "choose_action", None)
        if fn is None:
            raise RuntimeError(f"module strategy missing choose_action: {row['module_name']}")
        payload = fn(
            state=state,
            player=player,
            legal_actions=[{
                "player": a.player,
                "card_number": a.card_number,
                "surrender": a.surrender,
                "pass_turn": a.pass_turn,
                "use_sp_attack": a.use_sp_attack,
                "rotation": a.rotation,
                "x": a.x,
                "y": a.y,
            } for a in actions],
            context=row,
        )
        for a in actions:
            if (
                a.card_number == payload.get("card_number")
                and a.pass_turn == bool(payload.get("pass_turn", False))
                and a.use_sp_attack == bool(payload.get("use_sp_attack", False))
                and a.rotation == int(payload.get("rotation", 0))
                and a.x == payload.get("x")
                and a.y == payload.get("y")
            ):
                return a
        raise RuntimeError(f"module strategy returned non-legal action: {strategy_id}")

    raise ValueError(f"unsupported strategy kind: {row['kind']}")
