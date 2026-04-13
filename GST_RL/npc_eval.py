from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

if __package__:
    from .networks import PolicyValueNet
    from .rl_env import TableturfRLEnv
else:
    from networks import PolicyValueNet
    from rl_env import TableturfRLEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
NPC_JSON_CANDIDATES = [
    REPO_ROOT / "tableturf_sim" / "data" / "MiniGameGameNpcData.json",
    REPO_ROOT / "tableturf_sim" / "src" / "assets" / "MiniGameGameNpcData.json",
]

STYLE_MAP = {
    "Aggressive": "aggressive",
    "Balance": "balanced",
    "AccumulateSpecial": "conservative",
}
LEVEL_MAP = {0: "low", 1: "mid", 2: "high"}


def _load_npc() -> list[dict]:
    for p in NPC_JSON_CANDIDATES:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"NPC data missing: {[str(x) for x in NPC_JSON_CANDIDATES]}")


def _gyml_to_rowid(path: str) -> str:
    return path.split("/")[-1].split(".spl__")[0]


def npc_profile(npc_rowid: str, level: int) -> Dict[str, str]:
    arr = _load_npc()
    npc = next((x for x in arr if x.get("__RowId") == npc_rowid), None)
    if npc is None:
        all_ids = ", ".join(x.get("__RowId", "") for x in arr[:10])
        raise ValueError(f"NPC not found: {npc_rowid}. e.g. {all_ids}")

    levels = npc.get("AILevel", [])
    idx = next((i for i, lv in enumerate(levels) if int(lv) == int(level)), None)
    if idx is None:
        raise ValueError(f"NPC {npc_rowid} has no level={level}")

    ai_type = npc["AIType"][idx]
    map_id = npc["Map"][idx]
    deck_rowid = _gyml_to_rowid(npc["Deck"][idx])
    return {
        "npc_rowid": npc_rowid,
        "npc_name": npc.get("Name", npc_rowid),
        "map_id": map_id,
        "p2_deck": deck_rowid,
        "bot_style": STYLE_MAP[str(ai_type)],
        "bot_level": LEVEL_MAP[int(level)],
    }


def load_model(checkpoint: str, env: TableturfRLEnv, device: torch.device) -> PolicyValueNet:
    model = PolicyValueNet(
        map_channels=env.map_shape[0],
        scalar_dim=env.scalar_dim,
        action_feature_dim=env.action_feature_dim,
    ).to(device)
    obj = torch.load(checkpoint, map_location=device)
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_eval(model: PolicyValueNet, env: TableturfRLEnv, episodes: int, device: torch.device) -> Tuple[float, float, float]:
    wins = 0
    draws = 0
    losses = 0
    for _ in range(episodes):
        map_obs, scalar_obs, action_feats = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                logits, _ = model.forward_single(
                    torch.as_tensor(map_obs, dtype=torch.float32, device=device),
                    torch.as_tensor(scalar_obs, dtype=torch.float32, device=device),
                    torch.as_tensor(action_feats, dtype=torch.float32, device=device),
                )
                action_idx = int(torch.argmax(logits).item())
            step = env.step(action_idx)
            map_obs, scalar_obs, action_feats, done = step.map_obs, step.scalar_obs, step.action_features, step.done
        w = env.state.winner if env.state is not None else ""
        if w == "P1":
            wins += 1
        elif w == "draw":
            draws += 1
        else:
            losses += 1
    n = float(episodes)
    return wins / n, draws / n, losses / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model against one NPC profile.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--npc", type=str, required=True, help="NPC __RowId, e.g. MiniGame_WeaponShop")
    parser.add_argument("--npc-level", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--p1-deck", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    prof = npc_profile(args.npc, args.npc_level)
    env = TableturfRLEnv(
        map_id=prof["map_id"],
        p1_deck_selector=args.p1_deck,
        p2_deck_selector=prof["p2_deck"],
        bot_style=prof["bot_style"],
        bot_level=prof["bot_level"],
        seed=args.seed,
    )
    device = torch.device(args.device)
    model = load_model(args.checkpoint, env, device)
    win, draw, lose = run_eval(model, env, args.episodes, device)
    print(
        f"npc={prof['npc_rowid']}({prof['npc_name']}) level={args.npc_level} "
        f"map={prof['map_id']} p1_deck={env.p1_deck_rowid} p2_deck={prof['p2_deck']} "
        f"bot={prof['bot_style']}/{prof['bot_level']}"
    )
    print(f"episodes={args.episodes} win_rate={win:.3f} draw_rate={draw:.3f} loss_rate={lose:.3f}")


if __name__ == "__main__":
    main()

