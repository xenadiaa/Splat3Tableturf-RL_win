#!/usr/bin/env python3
"""One-time, idempotent backfill of Macro6 per-player statistics."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
STATS_PATH = BASE_DIR / "macro6_watchdog_stats.json"
ARCHIVE_ROOT = BASE_DIR / "pokopia_stamp_records"


PlayerCounts = dict[str, dict[str, int]]


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"[警告] 跳过无法解析的日志行：{path.name}:"
                        f"{line_number} ({exc})"
                    )
                    continue
                if isinstance(value, dict):
                    records.append(value)
    except OSError as exc:
        print(f"[警告] 无法读取 {path}：{exc}")
    return records


def _date_from_folder(path: Path) -> str:
    name = path.parent.name
    return name[6:] if name.startswith("STAMP_") else ""


def _increment(target: PlayerCounts, day: str, name: str, count: int) -> None:
    player_name = str(name).strip()
    amount = max(0, int(count))
    if not day or not player_name or amount <= 0:
        return
    players = target.setdefault(day, {})
    players[player_name] = players.get(player_name, 0) + amount


def _collect_reopen_summaries() -> tuple[
    PlayerCounts,
    PlayerCounts,
    PlayerCounts,
    PlayerCounts,
    PlayerCounts,
    int,
]:
    visits: PlayerCounts = {}
    tasks: PlayerCounts = {}
    returned_failures: PlayerCounts = {}
    room_closed_failures: PlayerCounts = {}
    network_error_failures: PlayerCounts = {}
    file_count = 0
    for path in sorted(ARCHIVE_ROOT.glob("STAMP_*/PLAYERS_*.jsonl")):
        day = _date_from_folder(path)
        if not day:
            continue
        file_count += 1
        for record in _read_json_lines(path):
            if record.get("status") != "room_reopen_summary":
                continue
            players = record.get("players", ())
            if not isinstance(players, list):
                continue
            for visit in players:
                if not isinstance(visit, dict):
                    continue
                name = str(visit.get("player_name", "")).strip()
                if visit.get("result") == "entry_failed":
                    failure_reason = str(visit.get("failure_reason") or "")
                    if failure_reason == "network_error_before_arrival":
                        target = network_error_failures
                    elif bool(visit.get("ended_by_room_freeze", False)):
                        target = room_closed_failures
                    else:
                        target = returned_failures
                    _increment(target, day, name, 1)
                    continue
                if visit.get("result") != "entered":
                    continue
                _increment(visits, day, name, 1)
                try:
                    completed = min(
                        3,
                        max(
                            0,
                            int(
                                visit.get(
                                    "available_reward_tasks",
                                    visit.get("completed_reward_tasks", 0),
                                )
                            ),
                        ),
                    )
                except (TypeError, ValueError):
                    completed = 0
                _increment(tasks, day, name, completed)
    return (
        visits,
        tasks,
        returned_failures,
        room_closed_failures,
        network_error_failures,
        file_count,
    )


def _collect_event_logs() -> tuple[
    PlayerCounts,
    PlayerCounts,
    PlayerCounts,
    PlayerCounts,
    PlayerCounts,
    int,
]:
    visits: PlayerCounts = {}
    tasks: PlayerCounts = {}
    returned_failures: PlayerCounts = {}
    room_closed_failures: PlayerCounts = {}
    network_error_failures: PlayerCounts = {}
    file_count = 0

    for path in sorted(ARCHIVE_ROOT.glob("STAMP_*/STAMP_*.jsonl")):
        day = _date_from_folder(path)
        if not day:
            continue
        file_count += 1
        active: dict[str, dict[str, int | bool]] = {}

        def finish(name: str, failure_reason: str = "") -> None:
            visit = active.pop(name, None)
            if not visit:
                return
            if not bool(visit.get("arrived", False)):
                if failure_reason == "left_before_arrival":
                    _increment(returned_failures, day, name, 1)
                elif failure_reason == "room_closed_before_arrival":
                    _increment(room_closed_failures, day, name, 1)
                elif failure_reason == "network_error_before_arrival":
                    _increment(network_error_failures, day, name, 1)
                return
            _increment(visits, day, name, 1)
            _increment(tasks, day, name, int(visit.get("tasks", 0)))

        def finish_all(failure_reason: str = "") -> None:
            for active_name in tuple(active):
                finish(active_name, failure_reason)

        for record in _read_json_lines(path):
            status = str(record.get("status", ""))

            # A saved CODE starts a new room.  The round-end screenshot is
            # emitted before the first reopen input and closes the old room.
            if status == "round_end_before_reopen" or (
                not status and str(record.get("code", "")).strip()
            ):
                finish_all(
                    "network_error_before_arrival"
                    if record.get("reason") == "network_connection_error"
                    else "room_closed_before_arrival"
                )
                continue

            if status == "player_room_status":
                name = str(record.get("player_name", "")).strip()
                notification = str(record.get("notification_status", ""))
                room_status = str(record.get("room_status", ""))
                if not name:
                    continue
                if notification == "即将抵达。" or room_status == "路上":
                    previous = active.get(name)
                    if previous and bool(previous.get("arrived", False)):
                        finish(name)
                    active.setdefault(name, {"arrived": False, "tasks": 0})
                elif notification == "已抵达。" or room_status == "到达":
                    visit = active.setdefault(
                        name,
                        {"arrived": False, "tasks": 0},
                    )
                    visit["arrived"] = True
                elif notification == "回去了。" or record.get("action") == "remove":
                    finish(name, "left_before_arrival")
                continue

            if status == "player_reward_progress":
                players = record.get("players", ())
                if not isinstance(players, list):
                    continue
                for player in players:
                    if not isinstance(player, dict):
                        continue
                    name = str(player.get("player_name", "")).strip()
                    if not name:
                        continue
                    try:
                        completed = min(
                            3,
                            max(
                                0,
                                int(
                                    player.get(
                                        "available_reward_tasks",
                                        player.get(
                                            "completed_reward_tasks",
                                            0,
                                        ),
                                    )
                                ),
                            ),
                        )
                    except (TypeError, ValueError):
                        completed = 0
                    if completed <= 0 and player.get("room_status") not in {
                        "路上",
                        "到达",
                    }:
                        continue
                    visit = active.setdefault(
                        name,
                        {
                            "arrived": player.get("room_status") == "到达",
                            "tasks": 0,
                        },
                    )
                    if player.get("room_status") == "到达":
                        visit["arrived"] = True
                    visit["tasks"] = max(int(visit.get("tasks", 0)), completed)

        # Include a currently open or historically unclosed visit.  This is
        # the exact gap the one-time bridge is intended to recover.
        finish_all()

    return (
        visits,
        tasks,
        returned_failures,
        room_closed_failures,
        network_error_failures,
        file_count,
    )


def _nested_counts(value: object) -> PlayerCounts:
    if not isinstance(value, dict):
        return {}
    result: PlayerCounts = {}
    for day, raw_players in value.items():
        if not isinstance(raw_players, dict):
            continue
        for name, raw_count in raw_players.items():
            try:
                _increment(result, str(day), str(name), int(raw_count))
            except (TypeError, ValueError):
                continue
    return result


def _merge_max(*sources: PlayerCounts) -> PlayerCounts:
    result: PlayerCounts = {}
    for source in sources:
        for day, players in source.items():
            destination = result.setdefault(day, {})
            for name, count in players.items():
                destination[name] = max(destination.get(name, 0), count)
    return {
        day: dict(sorted(players.items()))
        for day, players in sorted(result.items())
        if players
    }


def _totals(daily: PlayerCounts) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for players in daily.values():
        for name, count in players.items():
            result[name] += count
    return dict(sorted(result.items()))


def _sum_counts(counts: PlayerCounts) -> int:
    return sum(sum(players.values()) for players in counts.values())


def main() -> int:
    print("Macro6 历史玩家统计一次性补全")
    print("运行前请先关闭 watchdog，避免统计文件被同时写入。")
    print()
    try:
        confirmation = input("确认 watchdog 已关闭，输入 YES 继续：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消，未修改任何文件。")
        return 1
    if confirmation.upper() != "YES":
        print("已取消，未修改任何文件。")
        return 1
    print()
    if not STATS_PATH.is_file():
        print(f"[错误] 找不到永久统计文件：{STATS_PATH}")
        return 1
    if not ARCHIVE_ROOT.is_dir():
        print(f"[错误] 找不到历史日志目录：{ARCHIVE_ROOT}")
        return 1

    try:
        with STATS_PATH.open("r", encoding="utf-8-sig") as source:
            payload = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[错误] 无法读取永久统计文件：{exc}")
        return 1
    if not isinstance(payload, dict):
        print("[错误] 永久统计文件不是JSON对象。")
        return 1

    (
        summary_visits,
        summary_tasks,
        summary_returned_failures,
        summary_room_closed_failures,
        summary_network_error_failures,
        summary_files,
    ) = _collect_reopen_summaries()
    (
        event_visits,
        event_tasks,
        event_returned_failures,
        event_room_closed_failures,
        event_network_error_failures,
        event_files,
    ) = _collect_event_logs()
    daily_visits = _merge_max(
        _nested_counts(payload.get("daily_player_visits")),
        summary_visits,
        event_visits,
    )
    daily_tasks = _merge_max(
        _nested_counts(payload.get("daily_player_completed_tasks")),
        summary_tasks,
        event_tasks,
    )
    daily_returned_failures = _merge_max(
        _nested_counts(
            payload.get("daily_player_returned_before_arrival")
        ),
        summary_returned_failures,
        event_returned_failures,
    )
    daily_room_closed_failures = _merge_max(
        _nested_counts(
            payload.get("daily_player_room_closed_before_arrival")
        ),
        summary_room_closed_failures,
        event_room_closed_failures,
    )
    daily_network_error_failures = _merge_max(
        _nested_counts(
            payload.get("daily_player_network_error_before_arrival")
        ),
        summary_network_error_failures,
        event_network_error_failures,
    )

    total_visits = _totals(daily_visits)
    total_tasks = _totals(daily_tasks)
    total_returned_failures = _totals(daily_returned_failures)
    total_room_closed_failures = _totals(daily_room_closed_failures)
    total_network_error_failures = _totals(daily_network_error_failures)
    existing_daily_entries = payload.get("daily_player_entries", {})
    if not isinstance(existing_daily_entries, dict):
        existing_daily_entries = {}
    daily_entries: dict[str, int] = {}
    for day in set(existing_daily_entries) | set(daily_visits):
        try:
            existing = max(0, int(existing_daily_entries.get(day, 0)))
        except (TypeError, ValueError):
            existing = 0
        daily_entries[str(day)] = max(
            existing,
            sum(daily_visits.get(str(day), {}).values()),
        )
    existing_daily_failures = payload.get("daily_player_entry_failures", {})
    if not isinstance(existing_daily_failures, dict):
        existing_daily_failures = {}
    daily_failures: dict[str, int] = {}
    failure_days = (
        set(existing_daily_failures)
        | set(daily_returned_failures)
        | set(daily_room_closed_failures)
        | set(daily_network_error_failures)
    )
    for day in failure_days:
        try:
            existing = max(0, int(existing_daily_failures.get(day, 0)))
        except (TypeError, ValueError):
            existing = 0
        categorized = (
            sum(daily_returned_failures.get(str(day), {}).values())
            + sum(daily_room_closed_failures.get(str(day), {}).values())
            + sum(daily_network_error_failures.get(str(day), {}).values())
        )
        daily_failures[str(day)] = max(existing, categorized)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backup_path = STATS_PATH.with_name(
        f"{STATS_PATH.stem}.before_player_backfill_{timestamp}.bak"
    )
    report_path = ARCHIVE_ROOT / f"PLAYER_STATS_BACKFILL_{timestamp}.json"
    temporary_path = STATS_PATH.with_suffix(STATS_PATH.suffix + ".tmp")

    payload["version"] = max(9, int(payload.get("version", 0)))
    payload["daily_player_entries"] = dict(sorted(daily_entries.items()))
    payload["total_player_entries"] = max(
        int(payload.get("total_player_entries", 0)),
        sum(daily_entries.values()),
    )
    payload["daily_player_visits"] = daily_visits
    payload["total_player_visits"] = total_visits
    payload["daily_player_completed_tasks"] = daily_tasks
    payload["total_player_completed_tasks"] = total_tasks
    payload["daily_player_entry_failures"] = dict(
        sorted(daily_failures.items())
    )
    payload["total_player_entry_failures"] = max(
        int(payload.get("total_player_entry_failures", 0)),
        sum(daily_failures.values()),
    )
    payload["daily_player_returned_before_arrival"] = (
        daily_returned_failures
    )
    payload["total_player_returned_before_arrival"] = (
        total_returned_failures
    )
    payload["daily_player_room_closed_before_arrival"] = (
        daily_room_closed_failures
    )
    payload["total_player_room_closed_before_arrival"] = (
        total_room_closed_failures
    )
    payload["daily_player_network_error_before_arrival"] = (
        daily_network_error_failures
    )
    payload["total_player_network_error_before_arrival"] = (
        total_network_error_failures
    )
    payload["player_statistics_backfill"] = {
        "completed_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "method": "per-day per-player maximum of existing counters, room summaries, and event reconstruction",
        "summary_files_scanned": summary_files,
        "event_files_scanned": event_files,
        "recovered_player_visits": _sum_counts(daily_visits),
        "recovered_player_completed_tasks": _sum_counts(daily_tasks),
        "recovered_returned_before_arrival": _sum_counts(
            daily_returned_failures
        ),
        "recovered_room_closed_before_arrival": _sum_counts(
            daily_room_closed_failures
        ),
        "recovered_network_error_before_arrival": _sum_counts(
            daily_network_error_failures
        ),
        "backup": backup_path.name,
        "report": report_path.name,
    }
    report = {
        **payload["player_statistics_backfill"],
        "daily_player_visits": daily_visits,
        "total_player_visits": total_visits,
        "daily_player_completed_tasks": daily_tasks,
        "total_player_completed_tasks": total_tasks,
        "daily_player_returned_before_arrival": daily_returned_failures,
        "total_player_returned_before_arrival": total_returned_failures,
        "daily_player_room_closed_before_arrival": (
            daily_room_closed_failures
        ),
        "total_player_room_closed_before_arrival": (
            total_room_closed_failures
        ),
        "daily_player_network_error_before_arrival": (
            daily_network_error_failures
        ),
        "total_player_network_error_before_arrival": (
            total_network_error_failures
        ),
    }

    try:
        shutil.copy2(STATS_PATH, backup_path)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(STATS_PATH)
    except OSError as exc:
        print(f"[错误] 写入补全结果失败：{exc}")
        return 1

    print(f"已扫描每轮总结文件：{summary_files} 个")
    print(f"已扫描详细事件日志：{event_files} 个")
    print(f"补全后玩家来访记录：{_sum_counts(daily_visits)} 次")
    print(f"补全后玩家参与完成任务：{_sum_counts(daily_tasks)} 次")
    print(
        "补全后到达前返回："
        f"{_sum_counts(daily_returned_failures)} 次"
    )
    print(
        "补全后房间关闭时未抵达："
        f"{_sum_counts(daily_room_closed_failures)} 次"
    )
    print(
        "补全后网络连接错误时未抵达："
        f"{_sum_counts(daily_network_error_failures)} 次"
    )
    print(f"原统计备份：{backup_path}")
    print(f"补全报告：{report_path}")
    print("完成。现在可以使用两个统计查询CMD查看结果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
