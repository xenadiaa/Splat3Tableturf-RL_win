from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tableturf_sim.src.assets.types import Card_Single, Map_PointBit
except Exception:
    from tableturf_sim.src.assets.tableturf_types import Card_Single, Map_PointBit
from tableturf_sim.src.engine.env_core import Action, GameState, init_state, legal_actions, step


CARD_JSON_CANDIDATES = [
    REPO_ROOT / "tableturf_sim" / "data" / "cards" / "MiniGameCardInfo.json",
    REPO_ROOT / "tableturf_sim" / "src" / "assets" / "MiniGameCardInfo.json",
]
DECK_JSON_CANDIDATES = [
    REPO_ROOT / "tableturf_sim" / "data" / "cards" / "MiniGamePresetDeck.json",
    REPO_ROOT / "tableturf_sim" / "src" / "assets" / "MiniGamePresetDeck.json",
]
PLAYER_DECK_JSON_CANDIDATES = [
    REPO_ROOT / "tableturf_sim" / "data" / "cards" / "PlayerPresetDeck.json",
    REPO_ROOT / "tableturf_sim" / "src" / "assets" / "PlayerPresetDeck.json",
]
MAP_JSON_CANDIDATES = [
    REPO_ROOT / "tableturf_sim" / "data" / "maps" / "MiniGameMapInfo.json",
    REPO_ROOT / "tableturf_sim" / "src" / "assets" / "MiniGameMapInfo.json",
]
NPC_JSON_CANDIDATES = [
    REPO_ROOT / "tableturf_sim" / "data" / "MiniGameGameNpcData.json",
    REPO_ROOT / "tableturf_sim" / "src" / "assets" / "MiniGameGameNpcData.json",
]


def _gyml_to_rowid(path: str) -> str:
    return path.split("/")[-1].split(".spl__")[0]


def _read_json_from_candidates(paths: List[Path]) -> List[dict]:
    for path in paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"JSON not found in candidates: {[str(p) for p in paths]}")


def _load_card_number_index() -> Dict[str, int]:
    cards = _read_json_from_candidates(CARD_JSON_CANDIDATES)
    return {item["__RowId"]: int(item["Number"]) for item in cards}


def _load_preset_decks() -> List[dict]:
    return _read_json_from_candidates(DECK_JSON_CANDIDATES)


def _load_player_preset_decks() -> List[dict]:
    return _read_json_from_candidates(PLAYER_DECK_JSON_CANDIDATES)


def _load_npc_data() -> List[dict]:
    return _read_json_from_candidates(NPC_JSON_CANDIDATES)


def list_map_ids() -> List[str]:
    maps = _read_json_from_candidates(MAP_JSON_CANDIDATES)
    return [str(item["id"]) for item in maps]


def map_name_by_id(map_id: str) -> str:
    maps = _read_json_from_candidates(MAP_JSON_CANDIDATES)
    match = next((item for item in maps if str(item["id"]) == map_id), None)
    if match is None:
        valid = ", ".join(list_map_ids()[:8])
        raise ValueError(f"Unknown map id: {map_id}. e.g. {valid}")
    return str(match["name"])


def list_deck_rowids() -> List[str]:
    return [str(item["__RowId"]) for item in _load_preset_decks()]


def list_player_deck_names() -> List[str]:
    return [str(item.get("Name", "")) for item in _load_player_preset_decks()]


def player_map_deck_selector(map_id: str) -> str:
    map_name = map_name_by_id(map_id)
    names = set(list_player_deck_names())
    if map_name not in names:
        sample = ", ".join(list_player_deck_names()[:8])
        raise ValueError(f"No player preset deck matched map {map_id} -> {map_name}. e.g. {sample}")
    return f"player:{map_name}"


def map_deck_candidates(map_id: str) -> List[str]:
    counts: Dict[str, int] = {}
    for npc in _load_npc_data():
        maps = npc.get("Map", [])
        decks = npc.get("Deck", [])
        for m, deck_path in zip(maps, decks):
            if str(m) != map_id:
                continue
            rowid = _gyml_to_rowid(str(deck_path))
            counts[rowid] = counts.get(rowid, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [rowid for rowid, _ in ranked]


def auto_deck_rowids_for_map(map_id: str) -> Tuple[str, str]:
    candidates = map_deck_candidates(map_id)
    if not candidates:
        decks = list_deck_rowids()
        return decks[0], decks[1 if len(decks) > 1 else 0]
    p1 = candidates[0]
    p2 = candidates[1] if len(candidates) > 1 else candidates[0]
    return p1, p2


def level_npc_profiles_for_map(level: int, map_id: str) -> List[Dict[str, str]]:
    profiles: List[Dict[str, str]] = []
    for npc in _load_npc_data():
        rowid = str(npc.get("__RowId", ""))
        name = str(npc.get("Name", rowid))
        for ai_level, npc_map, ai_type, deck_path in zip(
            npc.get("AILevel", []),
            npc.get("Map", []),
            npc.get("AIType", []),
            npc.get("Deck", []),
        ):
            if int(ai_level) != int(level) or str(npc_map) != map_id:
                continue
            profiles.append(
                {
                    "npc_rowid": rowid,
                    "npc_name": name,
                    "map_id": str(npc_map),
                    "p2_deck": _gyml_to_rowid(str(deck_path)),
                    "bot_style_raw": str(ai_type),
                }
            )
    return profiles


def resolve_deck_numbers(selector: Optional[str], fallback_index: int = 0) -> List[int]:
    decks = _load_preset_decks()
    player_decks = _load_player_preset_decks()
    card_by_rowid = _load_card_number_index()

    if selector is None:
        chosen = decks[fallback_index % len(decks)]
        chosen_cards = chosen["Card"]
    elif selector.startswith("player:"):
        query = selector.split(":", 1)[1].strip()
        picked = next((d for d in player_decks if d.get("__RowId") == query), None)
        if picked is None:
            picked = next((d for d in player_decks if str(d.get("Name", "")) == query), None)
        if picked is None and query.isdigit():
            idx = int(query)
            if 0 <= idx < len(player_decks):
                picked = player_decks[idx]
        if picked is None:
            sample = ", ".join(list_player_deck_names()[:8])
            raise ValueError(f"Unknown player deck selector: {selector}. e.g. player:{sample}")
        chosen = {"__RowId": picked.get("__RowId", "PlayerDeck"), "Card": picked["Card"]}
        chosen_cards = picked["Card"]
    elif selector.isdigit():
        chosen = decks[int(selector) % len(decks)]
        chosen_cards = chosen["Card"]
    else:
        match = next((d for d in decks if d["__RowId"] == selector), None)
        if match is None:
            valid = ", ".join(list_deck_rowids()[:8])
            raise ValueError(f"Unknown deck selector: {selector}. e.g. {valid}")
        chosen = match
        chosen_cards = chosen["Card"]

    numbers: List[int] = []
    for card_path in chosen_cards:
        rowid = _gyml_to_rowid(card_path)
        numbers.append(card_by_rowid[rowid])
    if len(numbers) != 15:
        raise RuntimeError(f"Deck {chosen['__RowId']} does not contain 15 cards")
    return numbers


def _max_map_size_with_padding() -> Tuple[int, int]:
    maps = _read_json_from_candidates(MAP_JSON_CANDIDATES)
    widths = [len(item["point_type"][0]) for item in maps]
    heights = [len(item["point_type"]) for item in maps]
    return max(widths) + 8, max(heights) + 8


@dataclass
class EnvStep:
    map_obs: np.ndarray
    scalar_obs: np.ndarray
    action_features: np.ndarray
    done: bool
    reward: float
    info: Dict[str, object]


class TableturfRLEnv:
    """Single-agent env wrapper (agent=P1, opponent=P2 bot)."""

    def __init__(
        self,
        map_id: str = "ManySp",
        p1_deck_selector: Optional[str] = None,
        p2_deck_selector: Optional[str] = None,
        bot_style: str = "balanced",
        bot_level: str = "mid",
        seed: Optional[int] = 42,
    ) -> None:
        self.map_id = map_id
        auto_p1_rowid, auto_p2_rowid = auto_deck_rowids_for_map(self.map_id)
        p1_pick = p1_deck_selector or auto_p1_rowid
        p2_pick = p2_deck_selector or auto_p2_rowid
        self.p1_deck_rowid = p1_pick
        self.p2_deck_rowid = p2_pick
        self.p1_deck_ids = resolve_deck_numbers(p1_pick, fallback_index=0)
        self.p2_deck_ids = resolve_deck_numbers(p2_pick, fallback_index=1)
        self.bot_style = bot_style
        self.bot_level = bot_level
        self.base_seed = seed
        self._rng = random.Random(seed)
        self.max_w, self.max_h = _max_map_size_with_padding()
        self.state: Optional[GameState] = None
        self._cached_actions: List[Action] = []

    @property
    def action_dim(self) -> int:
        if self.state is None:
            return 0
        return len(self._cached_actions)

    @property
    def map_shape(self) -> Tuple[int, int, int]:
        return (6, self.max_h, self.max_w)

    @property
    def scalar_dim(self) -> int:
        return 6

    @property
    def action_feature_dim(self) -> int:
        return 12

    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        seed = self._rng.randint(0, 2**31 - 1) if self.base_seed is not None else None
        self.state = init_state(
            map_id=self.map_id,
            p1_deck_ids=self.p1_deck_ids,
            p2_deck_ids=self.p2_deck_ids,
            seed=seed,
            mode="1P",
            bot_style=self.bot_style,
            bot_level=self.bot_level,
            log_path="/dev/null",
        )
        map_obs, scalar_obs = self._encode_state()
        action_feats = self._encode_actions()
        return map_obs, scalar_obs, action_feats

    def step(self, action_index: int) -> EnvStep:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        if action_index < 0 or action_index >= len(self._cached_actions):
            raise IndexError(f"action index out of range: {action_index}")

        prev_diff = self._score_diff()
        action = self._cached_actions[action_index]
        ok, reason, result = step(self.state, action)

        if not ok:
            map_obs, scalar_obs = self._encode_state()
            action_feats = self._encode_actions()
            return EnvStep(
                map_obs=map_obs,
                scalar_obs=scalar_obs,
                action_features=action_feats,
                done=False,
                reward=-1.0,
                info={"ok": False, "reason": reason, "result": result},
            )

        done = bool(self.state.done)
        new_diff = self._score_diff()
        reward = (new_diff - prev_diff) / 10.0 - 0.01
        info: Dict[str, object] = {"ok": True, "reason": reason, "result": result}

        if done:
            if self.state.winner == "P1":
                reward += 1.0
            elif self.state.winner == "P2":
                reward -= 1.0
            info["winner"] = self.state.winner

        map_obs, scalar_obs = self._encode_state()
        action_feats = self._encode_actions() if not done else np.zeros((1, self.action_feature_dim), dtype=np.float32)
        return EnvStep(
            map_obs=map_obs,
            scalar_obs=scalar_obs,
            action_features=action_feats,
            done=done,
            reward=float(reward),
            info=info,
        )

    def _encode_state(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.state is None:
            raise RuntimeError("Environment not reset")

        obs = np.zeros((6, self.max_h, self.max_w), dtype=np.float32)
        game_map = self.state.map
        for y in range(game_map.height):
            for x in range(game_map.width):
                m = int(game_map.get_point(x, y))
                obs[0, y, x] = 1.0 if (m & int(Map_PointBit.IsValid)) else 0.0
                obs[1, y, x] = 1.0 if (m & int(Map_PointBit.IsP1)) else 0.0
                obs[2, y, x] = 1.0 if (m & int(Map_PointBit.IsP2)) else 0.0
                obs[3, y, x] = 1.0 if (m & int(Map_PointBit.IsSp)) else 0.0
                obs[4, y, x] = 1.0 if (m & int(Map_PointBit.IsSupplySp)) else 0.0
                obs[5, y, x] = 1.0 if ((m & int(Map_PointBit.IsP1)) and (m & int(Map_PointBit.IsP2))) else 0.0

        p1_score, p2_score = self._compute_scores()
        scalar = np.array(
            [
                self.state.turn / max(1, self.state.max_turns),
                self.state.players["P1"].sp / 20.0,
                self.state.players["P2"].sp / 20.0,
                (p1_score - p2_score) / 100.0,
                len(self.state.players["P1"].draw_pile) / 15.0,
                len(self.state.players["P2"].draw_pile) / 15.0,
            ],
            dtype=np.float32,
        )
        return obs, scalar

    def _encode_actions(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Environment not reset")

        self._cached_actions = legal_actions(self.state, "P1")
        if not self._cached_actions:
            return np.zeros((1, self.action_feature_dim), dtype=np.float32)

        hand = self.state.players["P1"].hand
        hand_index = {card.Number: idx for idx, card in enumerate(hand)}
        feats: List[np.ndarray] = []

        for action in self._cached_actions:
            card = next((c for c in hand if c.Number == action.card_number), None)
            if card is None:
                continue
            cell_count, sp_count = self._card_cell_stats(card, action.rotation)
            x_norm = (action.x / max(1, self.max_w - 1)) if action.x is not None else 0.0
            y_norm = (action.y / max(1, self.max_h - 1)) if action.y is not None else 0.0
            feats.append(
                np.array(
                    [
                        hand_index[card.Number] / 3.0,
                        1.0 if action.pass_turn else 0.0,
                        1.0 if action.use_sp_attack else 0.0,
                        action.rotation / 3.0,
                        x_norm,
                        y_norm,
                        card.CardPoint / 20.0,
                        card.SpecialCost / 10.0,
                        cell_count / 64.0,
                        sp_count / 64.0,
                        self.state.players["P1"].sp / 20.0,
                        self.state.turn / max(1, self.state.max_turns),
                    ],
                    dtype=np.float32,
                )
            )

        if len(feats) != len(self._cached_actions):
            raise RuntimeError("Failed to encode one or more legal actions")
        return np.stack(feats, axis=0)

    def _card_cell_stats(self, card: Card_Single, rotation: int) -> Tuple[int, int]:
        matrix = card.get_square_matrix(rotation)
        cell_count = 0
        sp_count = 0
        for row in matrix:
            for value in row:
                if value != 0:
                    cell_count += 1
                if value == 2:
                    sp_count += 1
        return cell_count, sp_count

    def _compute_scores(self) -> Tuple[int, int]:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        p1 = 0
        p2 = 0
        game_map = self.state.map
        for y in range(game_map.height):
            for x in range(game_map.width):
                m = int(game_map.get_point(x, y))
                if (m & int(Map_PointBit.IsValid)) == 0:
                    continue
                is_p1 = (m & int(Map_PointBit.IsP1)) != 0
                is_p2 = (m & int(Map_PointBit.IsP2)) != 0
                if is_p1 and not is_p2:
                    p1 += 1
                elif is_p2 and not is_p1:
                    p2 += 1
        return p1, p2

    def _score_diff(self) -> float:
        p1, p2 = self._compute_scores()
        return float(p1 - p2)
