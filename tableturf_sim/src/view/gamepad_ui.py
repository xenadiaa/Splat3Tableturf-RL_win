"""Terminal gamepad-like UI state machine for 1P Tableturf."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..assets.tableturf_types import Card_Single, GameMap, Map_PointBit
from ..engine.env_core import Action, GameState, step
from ..engine.loaders import MAP_PADDING
from ..utils.common_utils import _card_cells_on_map, create_card_from_id, validate_place_card_action
from ..utils.localization import lookup_card_name_zh


DIRECTION_KEYS = {"UP", "DOWN", "LEFT", "RIGHT"}
BUTTON_KEYS = {"A", "B", "X", "Y", "L"}
KEYBOARD_TO_PAD = {
    "Z": "A",
    "X": "B",
    "A": "Y",
    "S": "X",
    "Q": "L",
}

PHASE_CARD_GRID = "card_grid"
PHASE_SP_PICK = "sp_pick"
PHASE_PASS_PICK = "pass_pick"
PHASE_PLACE = "place"
PHASE_SURRENDER_CONFIRM = "surrender_confirm"
PHASE_WAITING_RESOLVE = "waiting_resolve"

# ANSI truecolor helpers (RGB)
CLR_RESET = "\033[0m"
RGB_P1_SP = (255, 192, 0)      # orange
RGB_P1_FILL = (255, 255, 0)    # yellow
RGB_P2_FILL = (0, 112, 192)    # deep blue
RGB_P2_SP = (0, 176, 240)      # light blue
RGB_CONFLICT = (128, 128, 128) # gray
RGB_EMPTY = (0, 0, 0)          # black
RGB_PREVIEW = (160, 255, 160)  # preview hint
RGB_PREVIEW_BAD = (255, 64, 64)  # preview blocked warning (red)
RGB_LINK = (255, 64, 255)      # anchor marker


def _rgb(text: str, rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m{text}{CLR_RESET}"


@dataclass
class DeckSnapshot:
    initial: List[int]
    used: List[int]
    hand: List[int]


def _find_local_view_rightbottom_self_special_anchor(
    game_map: GameMap,
    is_p1: bool,
    view_x0: int,
    view_y0: int,
    view_w: int,
    view_h: int,
    flip_180: bool,
) -> Tuple[int, int]:
    owner_bit = int(Map_PointBit.IsP1) if is_p1 else int(Map_PointBit.IsP2)
    special_bit = int(Map_PointBit.IsSp)
    candidates: List[Tuple[int, int]] = []
    for y in range(view_y0, view_y0 + view_h):
        for x in range(view_x0, view_x0 + view_w):
            cell = int(game_map.get_point(x, y))
            if (cell & owner_bit) != 0 and (cell & special_bit) != 0:
                lx = x - view_x0
                ly = y - view_y0
                if flip_180:
                    lx = view_w - 1 - lx
                    ly = view_h - 1 - ly
                candidates.append((lx, ly))
    if not candidates:
        return (view_w // 2, view_h // 2)
    if str(game_map.map_id) == "ManyHole":
        # Sticky_Thicket: special-case to use the left-side spawn point.
        candidates.sort(key=lambda p: (p[0], -p[1]))
        return candidates[0]
    candidates.sort(key=lambda p: (p[0], p[1]), reverse=True)
    return candidates[0]


def _card_name(card: Card_Single) -> str:
    zh = lookup_card_name_zh(card.name)
    return f"{card.Number}:{card.name}[{zh or '未找到中文'}]"


def _card_name_by_number(card_number: int) -> str:
    try:
        card = create_card_from_id(card_number)
        return _card_name(card)
    except Exception:
        return f"{card_number}"


def _card_total_cells(card: Card_Single) -> int:
    # Card shape cell count is rotation-invariant; use 0 degree.
    matrix = card.get_square_matrix(0)
    total = 0
    for row in matrix:
        for value in row:
            if value != 0:
                total += 1
    return total


def _card_is_even_square_anchor_invariant(card: Card_Single) -> bool:
    matrix = card.get_square_matrix(0)
    xs: List[int] = []
    ys: List[int] = []
    for y in range(8):
        for x in range(8):
            if matrix[y][x] != 0:
                xs.append(x)
                ys.append(y)
    if not xs:
        return False
    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1
    if width != height or (width % 2) != 0:
        return False
    lp0 = card.get_link_pos(0)
    return (
        card.get_link_pos(1) == lp0
        and card.get_link_pos(2) == lp0
        and card.get_link_pos(3) == lp0
    )


def _card_bounds(card: Card_Single, rotation: int) -> Tuple[int, int, int, int]:
    matrix = card.get_square_matrix(rotation)
    xs: List[int] = []
    ys: List[int] = []
    for y in range(8):
        for x in range(8):
            if matrix[y][x] != 0:
                xs.append(x)
                ys.append(y)
    if not xs:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs), max(ys))


def _render_card_matrix_0deg(card: Card_Single) -> List[str]:
    """Render 8x8 card shape at 0 degree. '*'=special, '■'=fill, ' '=empty."""
    matrix = card.get_square_matrix(0)
    out: List[str] = []
    for row in matrix:
        chars: List[str] = []
        for value in row:
            if value == 2:
                chars.append("*")
            elif value == 1:
                chars.append("■")
            else:
                chars.append(".")
        out.append("".join(chars))
    return out


def _render_all_hand_previews(
    hand: List[Card_Single],
    normal_placeable: set[int],
    sp_placeable: set[int],
) -> List[str]:
    """Render 4 card previews in 2x2 layout with compact spacing."""
    lines: List[str] = []
    cards: List[Optional[Card_Single]] = [hand[i] if i < len(hand) else None for i in range(4)]

    def _preview(c: Optional[Card_Single]) -> List[str]:
        if c is None:
            return ["........"] * 8
        return _render_card_matrix_0deg(c)

    def _meta(c: Optional[Card_Single]) -> str:
        if c is None:
            return "(empty)"
        n_ok = "N" if c.Number in normal_placeable else "-"
        s_ok = "S" if c.Number in sp_placeable else "-"
        return f"{_card_total_cells(c)}#{c.SpecialCost}#{n_ok}{s_ok}"

    p0 = _preview(cards[0])
    p1 = _preview(cards[1])
    p2 = _preview(cards[2])
    p3 = _preview(cards[3])

    for i in range(8):
        lines.append(f"{p0[i]}  {p1[i]}")
    lines.append(f"{_meta(cards[0])}  {_meta(cards[1])}")
    lines.append("")
    for i in range(8):
        lines.append(f"{p2[i]}  {p3[i]}")
    lines.append(f"{_meta(cards[2])}  {_meta(cards[3])}")
    return lines


def _render_grid_choices(
    hand: List[Card_Single],
    cursor: Optional[Tuple[int, int]],
    sp: int,
    normal_placeable: Optional[set[int]] = None,
    sp_placeable: Optional[set[int]] = None,
) -> List[str]:
    slots: List[str] = []
    normal_placeable = normal_placeable or set()
    sp_placeable = sp_placeable or set()
    for i in range(4):
        if i < len(hand):
            c = hand[i]
            n_ok = "N" if c.Number in normal_placeable else "-"
            s_ok = "S" if c.Number in sp_placeable else "-"
            total_points = _card_total_cells(c)
            sp_blocks = "■" * max(0, int(c.SpecialCost))
            sp_text = sp_blocks if sp_blocks else "0"
            slots.append(f"{total_points}#{sp_text}#{n_ok}{s_ok}")
        else:
            slots.append("(empty)")

    rows = [
        [slots[0], slots[1]],
        [slots[2], slots[3]],
        ["[跳过]", "[特殊攻击]"],
    ]

    cell_width = max(len(item) for row in rows for item in row) + 4

    lines: List[str] = []
    for y in range(3):
        row_text: List[str] = []
        for x in range(2):
            item = rows[y][x]
            if cursor is not None and (x, y) == cursor:
                token = f"> {item} <"
            else:
                token = f"  {item}  "
            row_text.append(token.ljust(cell_width))
        lines.append(" | ".join(row_text))

    affordable = [c.Number for c in hand if c.SpecialCost <= sp]
    lines.append(f"SP可用卡: {affordable if affordable else '无'}")
    lines.append("卡牌标记: N=可普通放置, S=可SP放置")
    return lines


def _is_valid_cell(mask: int) -> bool:
    return (mask & int(Map_PointBit.IsValid)) != 0


def _is_p1(mask: int) -> bool:
    return (mask & int(Map_PointBit.IsP1)) != 0


def _is_p2(mask: int) -> bool:
    return (mask & int(Map_PointBit.IsP2)) != 0


def _is_sp(mask: int) -> bool:
    return (mask & int(Map_PointBit.IsSp)) != 0


def _is_conflict(mask: int) -> bool:
    return _is_p1(mask) and _is_p2(mask)


def _owner_score(mask: int) -> int:
    if not _is_valid_cell(mask):
        return 0
    p1 = _is_p1(mask)
    p2 = _is_p2(mask)
    if p1 and not p2:
        return 1
    if p2 and not p1:
        return 2
    return 0


def _score_grid(game_map: GameMap, grid: Optional[List[List[int]]] = None) -> Tuple[int, int]:
    g = grid if grid is not None else game_map.grid
    p1 = 0
    p2 = 0
    for y in range(game_map.height):
        for x in range(game_map.width):
            owner = _owner_score(int(g[y][x]))
            if owner == 1:
                p1 += 1
            elif owner == 2:
                p2 += 1
    return p1, p2


def _simulate_apply_p1(game_map: GameMap, cells: List[Tuple[int, int, int]]) -> List[List[int]]:
    out = [row[:] for row in game_map.grid]
    for x, y, cell_type in cells:
        if not (0 <= x < game_map.width and 0 <= y < game_map.height):
            continue
        old = int(out[y][x])
        if not _is_valid_cell(old):
            continue
        new_mask = int(Map_PointBit.IsValid) | int(Map_PointBit.IsP1)
        if cell_type == 2:
            new_mask |= int(Map_PointBit.IsSp)
        out[y][x] = new_mask
    return out


def _overlay_preview_lines(
    game_map: GameMap,
    preview_cells: Optional[List[Tuple[int, int, int]]] = None,
    preview_valid: Optional[bool] = None,
    preview_use_sp_attack: Optional[bool] = None,
    link_anchor: Optional[Tuple[int, int]] = None,
    view_x0: int = 0,
    view_y0: int = 0,
    view_w: Optional[int] = None,
    view_h: Optional[int] = None,
    flip_180: bool = False,
) -> List[str]:
    overlay: Dict[Tuple[int, int], int] = {}
    if preview_cells:
        for x, y, cell_type in preview_cells:
            if not (0 <= x < game_map.width and 0 <= y < game_map.height):
                continue
            dx = x - view_x0
            dy = y - view_y0
            if not (0 <= dx < (view_w if view_w is not None else game_map.width)):
                continue
            if not (0 <= dy < (view_h if view_h is not None else game_map.height)):
                continue
            if flip_180:
                dx = (view_w if view_w is not None else game_map.width) - 1 - dx
                dy = (view_h if view_h is not None else game_map.height) - 1 - dy
            overlay[(dx, dy)] = cell_type

    lines: List[str] = []
    vw = view_w if view_w is not None else game_map.width
    vh = view_h if view_h is not None else game_map.height
    header = "   " + " ".join(f"{x:02d}" for x in range(vw))
    lines.append(header)

    # Fixed-width (2 chars) tokens for strict alignment across terminals.
    TOK_SQUARE = "[]"
    TOK_TRI = "/\\"
    TOK_PREV = "[]"
    TOK_ANCHOR = "<>"

    def _view_to_game(x: int, y: int) -> Tuple[int, int]:
        if not flip_180:
            return (x + view_x0, y + view_y0)
        return (
            view_x0 + (vw - 1 - x),
            view_y0 + (vh - 1 - y),
        )

    def _cell_is_occupied_on_map_layer(x: int, y: int) -> bool:
        m = int(game_map.get_point(x, y))
        # Occupied means already has owner bits (fill/special/conflict are all covered).
        return _is_p1(m) or _is_p2(m)

    def _cell_is_special_or_conflict_on_map_layer(x: int, y: int) -> bool:
        m = int(game_map.get_point(x, y))
        return _is_sp(m) or _is_conflict(m)

    for y in range(vh):
        row_out = [f"{y:02d}"]
        for x in range(vw):
            gx, gy = _view_to_game(x, y)
            # link_anchor stays backend-only (not rendered).
            if (x, y) in overlay:
                # preview priority:
                # 1) occupied rule -> red (highest)
                # 2) whole preview invalid -> gray all remaining preview cells
                # 3) normal preview -> green
                if preview_use_sp_attack:
                    is_red = _cell_is_special_or_conflict_on_map_layer(gx, gy)
                else:
                    is_red = _cell_is_occupied_on_map_layer(gx, gy)
                if is_red:
                    pv_color = RGB_PREVIEW_BAD
                elif preview_valid is False:
                    pv_color = RGB_CONFLICT
                else:
                    pv_color = RGB_PREVIEW
                if overlay[(x, y)] == 2:
                    row_out.append(_rgb(TOK_TRI, pv_color))
                else:
                    row_out.append(_rgb(TOK_PREV, pv_color))
                continue

            m = int(game_map.get_point(gx, gy))
            if not _is_valid_cell(m):
                row_out.append("  ")
            elif _is_conflict(m):
                row_out.append(_rgb(TOK_SQUARE, RGB_CONFLICT))
            elif _is_p1(m) and _is_sp(m):
                if (m & int(Map_PointBit.IsSupplySp)) != 0:
                    row_out.append(_rgb(TOK_TRI, RGB_P1_SP))
                else:
                    row_out.append(_rgb(TOK_SQUARE, RGB_P1_SP))
            elif _is_p2(m) and _is_sp(m):
                if (m & int(Map_PointBit.IsSupplySp)) != 0:
                    row_out.append(_rgb(TOK_TRI, RGB_P2_SP))
                else:
                    row_out.append(_rgb(TOK_SQUARE, RGB_P2_SP))
            elif _is_p1(m):
                row_out.append(_rgb(TOK_SQUARE, RGB_P1_FILL))
            elif _is_p2(m):
                row_out.append(_rgb(TOK_SQUARE, RGB_P2_FILL))
            else:
                row_out.append(_rgb(TOK_SQUARE, RGB_EMPTY))
        lines.append(" ".join(row_out))

    return lines


def _simulate_sp_delta(game_map: GameMap, cells: List[Tuple[int, int, int]], is_p1: bool = True) -> Tuple[int, int]:
    self_overwrite = 0
    enemy_overwrite = 0
    owner_bit = int(Map_PointBit.IsP1) if is_p1 else int(Map_PointBit.IsP2)
    enemy_bit = int(Map_PointBit.IsP2) if is_p1 else int(Map_PointBit.IsP1)
    special_bit = int(Map_PointBit.IsSp)
    conflict_mask = int(Map_PointBit.IsP1) | int(Map_PointBit.IsP2)

    for x, y, _ in cells:
        if not (0 <= x < game_map.width and 0 <= y < game_map.height):
            continue
        m = int(game_map.get_point(x, y))
        if (m & special_bit) != 0 or (m & conflict_mask) == conflict_mask:
            continue
        if (m & owner_bit) != 0:
            self_overwrite += 1
        elif (m & enemy_bit) != 0:
            enemy_overwrite += 1
    return self_overwrite, enemy_overwrite


def _anchor_bounds_for_card(card: Card_Single, rotation: int, game_map: GameMap) -> Tuple[int, int, int, int]:
    """Return valid anchor range so the rotated card never crosses map edges."""
    matrix = card.get_square_matrix(rotation)
    link_x, link_y = card.get_link_pos(rotation)

    min_dx = 999
    max_dx = -999
    min_dy = 999
    max_dy = -999
    for cy in range(8):
        for cx in range(8):
            if matrix[cy][cx] == 0:
                continue
            dx = cx - link_x
            dy = cy - link_y
            min_dx = min(min_dx, dx)
            max_dx = max(max_dx, dx)
            min_dy = min(min_dy, dy)
            max_dy = max(max_dy, dy)

    if min_dx == 999:
        return (0, 0, 0, 0)

    min_ax = -min_dx
    max_ax = (game_map.width - 1) - max_dx
    min_ay = -min_dy
    max_ay = (game_map.height - 1) - max_dy
    return (min_ax, max_ax, min_ay, max_ay)


def _clamp_anchor(card: Card_Single, rotation: int, game_map: GameMap, anchor: Tuple[int, int]) -> Tuple[int, int]:
    min_ax, max_ax, min_ay, max_ay = _anchor_bounds_for_card(card, rotation, game_map)
    x = max(min_ax, min(anchor[0], max_ax))
    y = max(min_ay, min(anchor[1], max_ay))
    return (x, y)


def _logical_overflow_for_anchor(
    card: Card_Single,
    anchor_logical: Tuple[int, int],
    rotation: int,
    logical_w: int,
    logical_h: int,
    view_x0: int,
    view_y0: int,
    flip_180: bool,
) -> Dict[str, int]:
    """Return overflow amount on each side in logical map coordinates."""
    ax, ay = anchor_logical
    if flip_180:
        ax = logical_w - 1 - ax
        ay = logical_h - 1 - ay
    ex = ax + view_x0
    ey = ay + view_y0
    if flip_180 and _card_is_even_square_anchor_invariant(card):
        left, top, right, bottom = _card_bounds(card, rotation)
        ex += left - right
        ey += bottom - top
    cells = _card_cells_on_map(card, ex, ey, rotation)
    if not cells:
        return {"left": 0, "right": 0, "top": 0, "bottom": 0}

    local_points: List[Tuple[int, int]] = []
    for x, y, _t in cells:
        lx = x - view_x0
        ly = y - view_y0
        if flip_180:
            lx = logical_w - 1 - lx
            ly = logical_h - 1 - ly
        local_points.append((lx, ly))

    min_x = min(x for x, _y in local_points)
    max_x = max(x for x, _y in local_points)
    min_y = min(y for _x, y in local_points)
    max_y = max(y for _x, y in local_points)

    return {
        "left": max(0, -min_x),
        "right": max(0, max_x - (logical_w - 1)),
        "top": max(0, -min_y),
        "bottom": max(0, max_y - (logical_h - 1)),
    }


def _anchor_bounds_for_card(card: Card_Single, rotation: int, game_map: GameMap) -> Tuple[int, int, int, int]:
    """Return valid anchor range so the rotated card never crosses map edges."""
    matrix = card.get_square_matrix(rotation)
    link_x, link_y = card.get_link_pos(rotation)

    min_dx = 999
    max_dx = -999
    min_dy = 999
    max_dy = -999
    for cy in range(8):
        for cx in range(8):
            if matrix[cy][cx] == 0:
                continue
            dx = cx - link_x
            dy = cy - link_y
            min_dx = min(min_dx, dx)
            max_dx = max(max_dx, dx)
            min_dy = min(min_dy, dy)
            max_dy = max(max_dy, dy)

    if min_dx == 999:
        return (0, 0, 0, 0)

    min_ax = -min_dx
    max_ax = (game_map.width - 1) - max_dx
    min_ay = -min_dy
    max_ay = (game_map.height - 1) - max_dy
    return (min_ax, max_ax, min_ay, max_ay)


def _clamp_anchor(card: Card_Single, rotation: int, game_map: GameMap, anchor: Tuple[int, int]) -> Tuple[int, int]:
    min_ax, max_ax, min_ay, max_ay = _anchor_bounds_for_card(card, rotation, game_map)
    x = max(min_ax, min(anchor[0], max_ax))
    y = max(min_ay, min(anchor[1], max_ay))
    return (x, y)


class TerminalGamepadUI:
    def __init__(self, state: GameState, local_player: str = "P1", submit_action_fn=None):
        self.state = state
        self.local_player = local_player
        self.enemy_player = "P2" if local_player == "P1" else "P1"
        self.submit_action_fn = submit_action_fn or step
        self.pad = MAP_PADDING
        # User-facing logical map excludes the padded tolerance border.
        if state.map.width > self.pad * 2 and state.map.height > self.pad * 2:
            self.logical_w = state.map.width - self.pad * 2
            self.logical_h = state.map.height - self.pad * 2
            self.view_x0 = self.pad
            self.view_y0 = self.pad
        else:
            self.logical_w = state.map.width
            self.logical_h = state.map.height
            self.view_x0 = 0
            self.view_y0 = 0

        self.phase = PHASE_CARD_GRID
        self.cursor = (0, 0)
        self.pick_index = 0
        self.pick_pool: List[Card_Single] = []

        self.initial_anchor = _find_local_view_rightbottom_self_special_anchor(
            state.map,
            is_p1=(self.local_player == "P1"),
            view_x0=self.view_x0,
            view_y0=self.view_y0,
            view_w=self.logical_w,
            view_h=self.logical_h,
            flip_180=(self.local_player == "P2"),
        )
        self.remembered_anchor: Optional[Tuple[int, int]] = None

        self.selected_card: Optional[Card_Single] = None
        self.use_sp_attack = False
        self.rotation = 0
        self.anchor = self.initial_anchor
        self.submitted_action: Optional[Action] = None
        self.submitted_card: Optional[Card_Single] = None

        self.last_message = ""
        self.last_turn_p1_action: Optional[dict] = None
        self.last_turn_p2_action: Optional[dict] = None
        self.surrender_choice_yes = False

    def _uses_flipped_view(self) -> bool:
        return self.local_player == "P2"

    def _engine_rotation_for_card(self, card: Optional[Card_Single]) -> int:
        if not self._uses_flipped_view():
            return self.rotation % 4
        # Even-square anchor-invariant cards already keep the same effective
        # link corner after the local-view board flip; applying an extra 180deg
        # offset would move the apparent link_pos to the opposite corner.
        if card is not None and _card_is_even_square_anchor_invariant(card):
            return self.rotation % 4
        return (self.rotation + 2) % 4

    def _display_rotation_deg(self) -> int:
        # Rotation is presented in local-view semantics.
        return (self.rotation % 4) * 90

    def _local_overflow_side_for_key(self, key: str) -> str:
        return {
            "LEFT": "left",
            "RIGHT": "right",
            "UP": "top",
            "DOWN": "bottom",
        }[key]

    def _anchor_to_engine_xy(
        self,
        anchor: Optional[Tuple[int, int]] = None,
        card: Optional[Card_Single] = None,
        engine_rotation: Optional[int] = None,
    ) -> Tuple[int, int]:
        ax, ay = anchor if anchor is not None else self.anchor
        if self._uses_flipped_view():
            ax = self.logical_w - 1 - ax
            ay = self.logical_h - 1 - ay
        ex = ax + self.view_x0
        ey = ay + self.view_y0
        if (
            self._uses_flipped_view()
            and card is not None
            and engine_rotation is not None
            and _card_is_even_square_anchor_invariant(card)
        ):
            left, top, right, bottom = _card_bounds(card, engine_rotation)
            ex += left - right
            ey += bottom - top
        return (ex, ey)

    def _deck_snapshot(self, player: str) -> DeckSnapshot:
        ps = self.state.players[player]
        hand_nums = [c.Number for c in ps.hand]
        used = [n for n in ps.deck_ids if n not in hand_nums and not any(d.Number == n for d in ps.draw_pile)]
        return DeckSnapshot(initial=list(ps.deck_ids), used=used, hand=hand_nums)

    def _preview_outcome(self) -> Tuple[Optional[Tuple[int, int]], Optional[str], Optional[Tuple[int, int, int]]]:
        if self.phase != PHASE_PLACE or self.selected_card is None:
            return None, None, None
        engine_rotation = self._engine_rotation_for_card(self.selected_card)
        ex, ey = self._anchor_to_engine_xy(card=self.selected_card, engine_rotation=engine_rotation)
        ok, reason, cells = validate_place_card_action(
            card=self.selected_card,
            game_map=self.state.map,
            anchor_x=ex,
            anchor_y=ey,
            rotation=engine_rotation,
            is_p1=(self.local_player == "P1"),
            use_sp_attack=self.use_sp_attack,
        )
        if not ok:
            return None, reason, None
        p1_now, p2_now = _score_grid(self.state.map)
        sim_grid = [row[:] for row in self.state.map.grid]
        owner_bit = int(Map_PointBit.IsP1) if self.local_player == "P1" else int(Map_PointBit.IsP2)
        for x, y, cell_type in cells:
            if not (0 <= x < self.state.map.width and 0 <= y < self.state.map.height):
                continue
            old = int(sim_grid[y][x])
            if not _is_valid_cell(old):
                continue
            new_mask = int(Map_PointBit.IsValid) | owner_bit
            if cell_type == 2:
                new_mask |= int(Map_PointBit.IsSp)
            sim_grid[y][x] = new_mask
        p1_new, p2_new = _score_grid(self.state.map, sim_grid)
        return (p1_new, p2_new), reason, (p1_new - p1_now, p2_new - p2_now, len(cells))

    def _has_any_place_for_card(self, card: Card_Single, use_sp_attack: bool) -> bool:
        if use_sp_attack and self.state.players[self.local_player].sp < card.SpecialCost:
            return False
        for rot in (0, 1, 2, 3):
            for y in range(self.logical_h):
                for x in range(self.logical_w):
                    ok, _reason, _cells = validate_place_card_action(
                        card=card,
                        game_map=self.state.map,
                        anchor_x=self._anchor_to_engine_xy(
                            anchor=(x, y),
                            card=card,
                            engine_rotation=(
                                rot
                                if (not self._uses_flipped_view() or _card_is_even_square_anchor_invariant(card))
                                else (rot + 2) % 4
                            ),
                        )[0],
                        anchor_y=self._anchor_to_engine_xy(
                            anchor=(x, y),
                            card=card,
                            engine_rotation=(
                                rot
                                if (not self._uses_flipped_view() or _card_is_even_square_anchor_invariant(card))
                                else (rot + 2) % 4
                            ),
                        )[1],
                        rotation=(
                            rot
                            if (not self._uses_flipped_view() or _card_is_even_square_anchor_invariant(card))
                            else (rot + 2) % 4
                        ),
                        is_p1=(self.local_player == "P1"),
                        use_sp_attack=use_sp_attack,
                    )
                    if ok:
                        return True
        return False

    def _placeable_sets_for_hand(self) -> Tuple[set[int], set[int]]:
        normal_ok: set[int] = set()
        sp_ok: set[int] = set()
        for c in self.state.players[self.local_player].hand:
            if self._has_any_place_for_card(c, use_sp_attack=False):
                normal_ok.add(c.Number)
            if self._has_any_place_for_card(c, use_sp_attack=True):
                sp_ok.add(c.Number)
        return normal_ok, sp_ok

    def _render_right_panel(self) -> List[str]:
        lines = ["[右侧出牌信息]"]
        shown_card = self.selected_card if self.selected_card is not None else self.submitted_card
        if shown_card is not None:
            lines.append(f"己方当前(正面): {_card_name(shown_card)}")
            lines.append("对手当前(背面): [CARD BACK]")
        else:
            lines.append("己方当前(正面): (未选择)")
            lines.append("对手当前(背面): (等待)")

        my_last = self.last_turn_p1_action if self.local_player == "P1" else self.last_turn_p2_action
        enemy_last = self.last_turn_p2_action if self.local_player == "P1" else self.last_turn_p1_action
        if my_last:
            n1 = my_last.get("card_number")
            lines.append(f"上回合己方: {_card_name_by_number(int(n1)) if n1 is not None else 'PASS'}")
        if enemy_last:
            lines.append("上回合对手: [CARD BACK]")
        return lines

    def render(self) -> str:
        p1 = self.state.players[self.local_player]
        p2 = self.state.players[self.enemy_player]
        p1_score, p2_score = _score_grid(self.state.map)

        lines: List[str] = []
        lines.append(f"[对手] {self.enemy_player}  SP={p2.sp}")
        lines.append(f"[我方] {self.local_player}  SP={p1.sp}")
        if self.local_player == "P1":
            my_score, opp_score = p1_score, p2_score
        else:
            my_score, opp_score = p2_score, p1_score
        lines.append(f"回合 {self.state.turn}/{self.state.max_turns}  当前比分 我方:{my_score} 对手:{opp_score}")
        lines.append(f"阶段: {self.phase}")

        preview_cells: Optional[List[Tuple[int, int, int]]] = None
        preview_valid: Optional[bool] = None
        preview_use_sp_attack: Optional[bool] = None
        link_anchor: Optional[Tuple[int, int]] = None

        if self.phase == PHASE_CARD_GRID:
            normal_ok, sp_ok = self._placeable_sets_for_hand()
            lines.append("\n[左侧 2x2手牌 + 两按钮]")
            lines.extend(_render_grid_choices(p1.hand, self.cursor, p1.sp, normal_placeable=normal_ok, sp_placeable=sp_ok))
            idx = self._cursor_card_index()
            if idx is not None and idx < len(p1.hand):
                lines.append(f"卡牌名称: {_card_name(p1.hand[idx])}")

        elif self.phase == PHASE_WAITING_RESOLVE:
            normal_ok, sp_ok = self._placeable_sets_for_hand()
            lines.append("\n[等待对手确认]")
            lines.extend(_render_grid_choices(p1.hand, None, p1.sp, normal_placeable=normal_ok, sp_placeable=sp_ok))
            lines.append("已提交本回合动作，等待双方结算...")

        elif self.phase in (PHASE_SP_PICK, PHASE_PASS_PICK):
            pick_title = "特殊攻击选卡" if self.phase == PHASE_SP_PICK else "跳过选卡"
            lines.append(f"\n[{pick_title}] 左右选择，A确认，B返回")
            if not self.pick_pool:
                lines.append("无可选卡")
            for i, c in enumerate(self.pick_pool):
                marker = ">" if i == self.pick_index else " "
                extra = f" (sp={c.SpecialCost})" if self.phase == PHASE_SP_PICK else ""
                lines.append(f"{marker} {_card_name(c)}{extra}")

        elif self.phase == PHASE_PLACE and self.selected_card is not None:
            lines.append("\n[放置阶段] 方向键移动link-pos, X顺时针, Y逆时针, A确认, B取消")
            lines.append(f"当前卡: {_card_name(self.selected_card)}")
            lines.append(f"模式: {'SP攻击' if self.use_sp_attack else '普通放置'}")
            lines.append(f"rotation={self._display_rotation_deg()}°")
            engine_rotation = self._engine_rotation_for_card(self.selected_card)
            ex, ey = self._anchor_to_engine_xy(card=self.selected_card, engine_rotation=engine_rotation)
            ok, reason, cells = validate_place_card_action(
                card=self.selected_card,
                game_map=self.state.map,
                anchor_x=ex,
                anchor_y=ey,
                rotation=engine_rotation,
                is_p1=(self.local_player == "P1"),
                use_sp_attack=self.use_sp_attack,
            )
            lines.append(f"可放置: {ok} ({reason})")
            preview_cells = _card_cells_on_map(self.selected_card, ex, ey, engine_rotation)
            preview_valid = ok
            preview_use_sp_attack = self.use_sp_attack
            link_anchor = None

            predicted, _, deltas = self._preview_outcome()
            if predicted and deltas:
                dp1, dp2, _ = deltas
                lines.append(
                    f"预估落子后比分: P1 {predicted[0]} ({dp1:+d})  vs  P2 {predicted[1]} ({dp2:+d})"
                )
            else:
                lines.append("预估落子后比分: 无效位置，无法计算")

            # Keep SP overwrite hint for both normal/SP placement as requested
            self_delta, enemy_delta = _simulate_sp_delta(self.state.map, cells, is_p1=(self.local_player == "P1"))
            lines.append(f"覆盖预估: 敌方被覆盖 {enemy_delta}格, 己方被覆盖 {self_delta}格")
        elif self.phase == PHASE_SURRENDER_CONFIRM:
            yes = "> 是 <" if self.surrender_choice_yes else "  是  "
            no = "> 否 <" if not self.surrender_choice_yes else "  否  "
            lines.append("\n[投降确认] 左右选择，A确认，B取消")
            lines.append(f"{yes}   |   {no}")

        if self.phase == PHASE_WAITING_RESOLVE and self.submitted_action is not None:
            if not self.submitted_action.pass_turn and self.submitted_card is not None:
                ex = int(self.submitted_action.x or 0)
                ey = int(self.submitted_action.y or 0)
                preview_cells = _card_cells_on_map(self.submitted_card, ex, ey, self.submitted_action.rotation)
                preview_valid = True
                preview_use_sp_attack = self.submitted_action.use_sp_attack
            else:
                preview_cells = None
                preview_valid = None
                preview_use_sp_attack = None

        lines.append("\n" + "\n".join(self._render_right_panel()))
        lines.append("\n[地图预览]")
        lines.extend(
            _overlay_preview_lines(
                self.state.map,
                preview_cells=preview_cells,
                preview_valid=preview_valid,
                preview_use_sp_attack=preview_use_sp_attack,
                link_anchor=link_anchor,
                view_x0=self.view_x0,
                view_y0=self.view_y0,
                view_w=self.logical_w,
                view_h=self.logical_h,
                flip_180=self._uses_flipped_view(),
            )
        )

        lines.append("\n[L] 查看牌组状态")
        if self.last_message:
            lines.append(f"提示: {self.last_message}")
        return "\n".join(lines)

    def render_deck_page(self) -> str:
        s1 = self._deck_snapshot(self.local_player)
        s2 = self._deck_snapshot(self.enemy_player)
        undrawn_self = [c.Number for c in self.state.players[self.local_player].draw_pile]
        lines = ["[牌组状态页]"]
        lines.append(f"己方 总卡组: {s1.initial}")
        lines.append(f"己方 已出: {s1.used}")
        lines.append(f"己方 手牌: {s1.hand}")
        lines.append(f"己方 未抽到: {undrawn_self}")
        lines.append(f"对手 总卡组数量: {len(s2.initial)}")
        lines.append(f"对手 手牌数量: {len(self.state.players[self.enemy_player].hand)}")
        lines.append(f"对手 已出: {s2.used}")
        lines.append(f"对手 未抽到数量: {len(self.state.players[self.enemy_player].draw_pile)}")
        return "\n".join(lines)

    def _cursor_card_index(self) -> Optional[int]:
        x, y = self.cursor
        if y > 1:
            return None
        return y * 2 + x

    def _cursor_is_special_button(self) -> bool:
        return self.cursor == (1, 2)

    def _cursor_is_pass_button(self) -> bool:
        return self.cursor == (0, 2)

    def _enter_place_phase(self, card: Card_Single, use_sp_attack: bool) -> None:
        self.selected_card = card
        self.use_sp_attack = use_sp_attack
        self.rotation = 0
        self.anchor = self.remembered_anchor if self.remembered_anchor is not None else self.initial_anchor
        self.anchor = (
            max(0, min(self.logical_w - 1, self.anchor[0])),
            max(0, min(self.logical_h - 1, self.anchor[1])),
        )
        self.phase = PHASE_PLACE
        self.last_message = f"进入放置：{_card_name(card)}"

    def _on_turn_resolved(self, payload: dict) -> None:
        self.last_turn_p1_action = payload.get("p1_action") if isinstance(payload, dict) else None
        self.last_turn_p2_action = payload.get("p2_action") if isinstance(payload, dict) else None
        self.phase = PHASE_CARD_GRID
        self.cursor = (0, 0)
        self.pick_index = 0
        self.pick_pool = []
        self.selected_card = None
        self.use_sp_attack = False
        self.rotation = 0
        self.submitted_action = None
        self.submitted_card = None

    def _commit_action(self) -> bool:
        assert self.selected_card is not None
        submitting_card = self.selected_card
        ex, ey = self._anchor_to_engine_xy(card=submitting_card, engine_rotation=self._engine_rotation_for_card(submitting_card))
        action = Action(
            player=self.local_player,
            card_number=submitting_card.Number,
            pass_turn=False,
            use_sp_attack=self.use_sp_attack,
            rotation=self._engine_rotation_for_card(submitting_card),
            x=ex,
            y=ey,
        )
        ok, reason, payload = self.submit_action_fn(self.state, action)
        if not ok:
            self.last_message = f"提交失败: {reason}"
            return False
        self.submitted_action = action
        self.submitted_card = submitting_card
        if reason == "TURN_RESOLVED" and isinstance(payload, dict):
            self._on_turn_resolved(payload)
        else:
            self.phase = PHASE_WAITING_RESOLVE
            self.selected_card = None
            self.pick_pool = []
            self.pick_index = 0
            self.last_message = "已提交动作，等待对手..."
        self.last_message = f"回合动作已确认: {reason}" if reason == "TURN_RESOLVED" else "已提交动作，等待对手..."
        return True

    def _commit_pass(self, card: Card_Single) -> bool:
        action = Action(player=self.local_player, card_number=card.Number, pass_turn=True)
        ok, reason, payload = self.submit_action_fn(self.state, action)
        if not ok:
            self.last_message = f"跳过失败: {reason}"
            return False
        self.submitted_action = action
        self.submitted_card = card
        if reason == "TURN_RESOLVED" and isinstance(payload, dict):
            self._on_turn_resolved(payload)
            self.last_message = f"跳过已确认: {reason}"
        else:
            self.phase = PHASE_WAITING_RESOLVE
            self.pick_pool = []
            self.pick_index = 0
            self.selected_card = None
            self.last_message = "已提交跳过，等待对手..."
        return True

    def _commit_surrender(self) -> bool:
        action = Action(player=self.local_player, surrender=True)
        ok, reason, payload = self.submit_action_fn(self.state, action)
        if not ok:
            self.last_message = f"投降失败: {reason}"
            return False
        self.phase = PHASE_CARD_GRID
        self.last_message = f"投降已确认: {reason} winner={payload.get('winner')}"
        return True

    def handle_key(self, key: str) -> Optional[str]:
        key = key.strip().upper()
        key = KEYBOARD_TO_PAD.get(key, key)
        if key == "=":
            key = "+"
        if not key:
            return None

        if self.phase == PHASE_WAITING_RESOLVE:
            if key == "L":
                return self.render_deck_page()
            self.last_message = "已提交动作，等待双方结算"
            return None

        if key == "+" and self.phase != PHASE_SURRENDER_CONFIRM:
            self.phase = PHASE_SURRENDER_CONFIRM
            self.surrender_choice_yes = False
            return None

        if key == "L":
            return self.render_deck_page()

        if self.phase == PHASE_SURRENDER_CONFIRM:
            if key in ("LEFT", "RIGHT"):
                self.surrender_choice_yes = not self.surrender_choice_yes
                return None
            if key == "B":
                self.phase = PHASE_CARD_GRID
                self.last_message = "取消投降"
                return None
            if key == "A":
                if self.surrender_choice_yes:
                    self._commit_surrender()
                else:
                    self.phase = PHASE_CARD_GRID
                    self.last_message = "未投降，继续对局"
                return None
            self.last_message = "投降确认仅支持 左右/A/B"
            return None

        if self.phase == PHASE_CARD_GRID:
            if key in DIRECTION_KEYS:
                x, y = self.cursor
                if key == "UP":
                    y = max(0, y - 1)
                elif key == "DOWN":
                    y = min(2, y + 1)
                elif key == "LEFT":
                    x = max(0, x - 1)
                elif key == "RIGHT":
                    x = min(1, x + 1)
                self.cursor = (x, y)
                return None

            if key != "A":
                self.last_message = "卡牌选择阶段仅支持 方向键/A/L"
                return None

            p1 = self.state.players[self.local_player]
            if self._cursor_is_special_button():
                affordable = [c for c in p1.hand if c.SpecialCost <= p1.sp and self._has_any_place_for_card(c, use_sp_attack=True)]
                if not affordable:
                    self.last_message = "SP模式无可选卡（可能SP不足或无合法放置位置）"
                    return None
                self.phase = PHASE_SP_PICK
                self.pick_pool = affordable
                self.pick_index = 0
                return None

            if self._cursor_is_pass_button():
                self.phase = PHASE_PASS_PICK
                self.pick_pool = list(p1.hand)
                self.pick_index = 0
                return None

            idx = self._cursor_card_index()
            if idx is None or idx >= len(p1.hand):
                self.last_message = "该位置无卡牌"
                return None
            card = p1.hand[idx]
            if not self._has_any_place_for_card(card, use_sp_attack=False):
                self.last_message = "该卡当前无合法普通放置位置，不可选"
                return None
            self._enter_place_phase(card, use_sp_attack=False)
            return None

        if self.phase in (PHASE_SP_PICK, PHASE_PASS_PICK):
            if key == "B":
                self.phase = PHASE_CARD_GRID
                self.pick_pool = []
                self.pick_index = 0
                return None
            if key in ("LEFT", "UP"):
                if self.pick_pool:
                    self.pick_index = (self.pick_index - 1) % len(self.pick_pool)
                return None
            if key in ("RIGHT", "DOWN"):
                if self.pick_pool:
                    self.pick_index = (self.pick_index + 1) % len(self.pick_pool)
                return None
            if key == "A":
                if not self.pick_pool:
                    self.last_message = "无可选卡"
                    return None
                card = self.pick_pool[self.pick_index]
                if self.phase == PHASE_SP_PICK:
                    self._enter_place_phase(card, use_sp_attack=True)
                else:
                    self._commit_pass(card)
                self.pick_pool = []
                self.pick_index = 0
                return None
            self.last_message = "选卡阶段仅支持 左右/A/B/L"
            return None

        if self.phase == PHASE_PLACE:
            if key in DIRECTION_KEYS:
                x, y = self.anchor
                if key == "UP":
                    y -= 1
                elif key == "DOWN":
                    y += 1
                elif key == "LEFT":
                    x -= 1
                elif key == "RIGHT":
                    x += 1
                cand = (
                    max(0, min(self.logical_w - 1, x)),
                    max(0, min(self.logical_h - 1, y)),
                )
                if self.selected_card is not None:
                    cur_of = _logical_overflow_for_anchor(
                        self.selected_card,
                        self.anchor,
                        self._engine_rotation_for_card(self.selected_card),
                        self.logical_w,
                        self.logical_h,
                        self.view_x0,
                        self.view_y0,
                        self._uses_flipped_view(),
                    )
                    new_of = _logical_overflow_for_anchor(
                        self.selected_card,
                        cand,
                        self._engine_rotation_for_card(self.selected_card),
                        self.logical_w,
                        self.logical_h,
                        self.view_x0,
                        self.view_y0,
                        self._uses_flipped_view(),
                    )
                    local_side = self._local_overflow_side_for_key(key)
                    blocked = (
                        new_of[local_side] > cur_of[local_side]
                    )
                    if not blocked:
                        self.anchor = cand
                else:
                    self.anchor = cand
                return None
            if key == "X":
                self.rotation = (self.rotation + 1) % 4
                return None
            if key == "Y":
                self.rotation = (self.rotation - 1) % 4
                return None
            if key == "B":
                self.remembered_anchor = self.anchor
                self.phase = PHASE_CARD_GRID
                self.selected_card = None
                self.use_sp_attack = False
                self.rotation = 0
                self.last_message = f"取消放置，记录link-pos={self.remembered_anchor}"
                return None
            if key == "A":
                if self.selected_card is None:
                    self.last_message = "未选择卡牌"
                    return None
                self._commit_action()
                return None
            self.last_message = "放置阶段仅支持 方向键/A/B/X/Y/L"
        return None
