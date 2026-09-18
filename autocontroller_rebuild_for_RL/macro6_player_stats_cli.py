#!/usr/bin/env python3
"""Read Macro6's persistent player counters without modifying them."""

from __future__ import annotations

import argparse
import calendar
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re


STATS_PATH = Path(__file__).resolve().with_name("macro6_watchdog_stats.json")
BEIJING_TIMEZONE = timezone(timedelta(hours=8))


def _load_stats() -> dict[str, object]:
    if not STATS_PATH.is_file():
        raise FileNotFoundError(f"找不到统计文件：{STATS_PATH}")
    with STATS_PATH.open("r", encoding="utf-8-sig") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("统计文件的根内容不是 JSON 对象。")
    return payload


def _counter_map(payload: dict[str, object], key: str) -> dict[str, int]:
    raw = payload.get(key, {})
    if not isinstance(raw, dict):
        return {}
    result: dict[str, int] = {}
    for day, value in raw.items():
        try:
            result[str(day)] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    return result


def _nested_counter_map(
    payload: dict[str, object],
    key: str,
) -> dict[str, dict[str, int]]:
    raw = payload.get(key, {})
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict[str, int]] = {}
    for day, player_counts in raw.items():
        if not isinstance(player_counts, dict):
            continue
        result[str(day)] = {
            str(name): max(0, int(count))
            for name, count in player_counts.items()
            if str(name).strip()
        }
    return result


def _sorted_player_counts(counts: dict[str, int]) -> list[tuple[str, int]]:
    """Sort descending by count, then ascending by name for stable ties."""
    return sorted(
        ((name, count) for name, count in counts.items() if count > 0),
        key=lambda item: (-item[1], item[0].casefold(), item[0]),
    )


def _combine_player_counts(*sources: dict[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for source in sources:
        for name, count in source.items():
            result[name] = result.get(name, 0) + max(0, int(count))
    return result


def _sum_nested_range(
    payload: dict[str, object],
    key: str,
    start_day: date,
    end_day: date,
) -> dict[str, int]:
    daily = _nested_counter_map(payload, key)
    selected: list[dict[str, int]] = []
    for day, counts in daily.items():
        try:
            parsed_day = datetime.strptime(day, "%Y-%m-%d").date()
        except ValueError:
            continue
        if start_day <= parsed_day < end_day:
            selected.append(counts)
    return _combine_player_counts(*selected)


def _sum_daily_range(
    payload: dict[str, object],
    key: str,
    start_day: date,
    end_day: date,
) -> int:
    daily = _counter_map(payload, key)
    total = 0
    for day, count in daily.items():
        try:
            parsed_day = datetime.strptime(day, "%Y-%m-%d").date()
        except ValueError:
            continue
        if start_day <= parsed_day < end_day:
            total += count
    return total


def _show_daily_service_table(
    payload: dict[str, object],
    start_day: date,
    end_day: date,
) -> None:
    """Show recorded business days in a range and finish with range totals."""
    runs = _counter_map(payload, "daily_runs")
    successes = _counter_map(payload, "daily_player_entries")
    failures = _counter_map(payload, "daily_player_entry_failures")
    selected_days = []
    for raw_day in set(runs) | set(successes) | set(failures):
        try:
            parsed_day = datetime.strptime(raw_day, "%Y-%m-%d").date()
        except ValueError:
            continue
        if start_day <= parsed_day < end_day:
            selected_days.append(raw_day)
    selected_days.sort()

    print()
    print("每日服务人次：")
    print(
        f"{'业务日期':<12} {'成功服务人次':>12} "
        f"{'进入失败人次':>12}"
    )
    print("-" * 42)
    for day in selected_days:
        print(
            f"{day:<12} {successes.get(day, 0):>12} "
            f"{failures.get(day, 0):>12}"
        )
    if not selected_days:
        print("（该范围暂无每日玩家统计记录）")
    print("-" * 42)
    print(
        f"{'总计':<12} "
        f"{sum(successes.get(day, 0) for day in selected_days):>12} "
        f"{sum(failures.get(day, 0) for day in selected_days):>12}"
    )


def _show_player_ranking(
    title: str,
    counts: dict[str, int],
    action: str,
) -> None:
    print()
    print(title)
    ranking = _sorted_player_counts(counts)
    if not ranking:
        print("（暂无记录）")
        return
    for player_name, count in ranking:
        print(f"{player_name}，{action}{count}次")


def _show_failure_ranking(
    title: str,
    returned: dict[str, int],
    room_closed: dict[str, int],
    network_error: dict[str, int],
    aggregate_failure_count: int,
) -> None:
    print()
    print(title)
    combined = _combine_player_counts(returned, room_closed, network_error)
    ranking = _sorted_player_counts(combined)
    if not ranking:
        print("（暂无可归属到玩家的记录）")
    else:
        for player_name, total in ranking:
            print(
                f"{player_name}，进入失败{total}次"
                f"（到达前返回 {returned.get(player_name, 0)}次，"
                "房间关闭时未抵达 "
                f"{room_closed.get(player_name, 0)}次，"
                "网络连接错误时未抵达 "
                f"{network_error.get(player_name, 0)}次）"
            )
    unattributed = max(0, int(aggregate_failure_count) - sum(combined.values()))
    if unattributed:
        print(f"旧日志中无法归属到具体玩家/原因：{unattributed}次")


def _parse_business_day(raw: str) -> str:
    compact = raw.strip()
    if not re.fullmatch(r"\d{8}", compact):
        raise ValueError("日期必须是8位数字，例如 20260904。")
    parsed = datetime.strptime(compact, "%Y%m%d")
    return parsed.strftime("%Y-%m-%d")


def _previous_business_day() -> str:
    """Return the business day before the current Beijing 05:00 day."""
    now_beijing = datetime.now(BEIJING_TIMEZONE)
    current_business_day = (now_beijing - timedelta(hours=5)).date()
    return (current_business_day - timedelta(days=1)).isoformat()


def _current_business_day():
    return (datetime.now(BEIJING_TIMEZONE) - timedelta(hours=5)).date()


def _parse_month(raw: str):
    compact = raw.strip()
    if not re.fullmatch(r"\d{6}", compact):
        raise ValueError("年月必须是6位数字，例如 202609。")
    return datetime.strptime(compact, "%Y%m").date().replace(day=1)


def _parse_year(raw: str) -> int:
    compact = raw.strip()
    if not re.fullmatch(r"\d{4}", compact):
        raise ValueError("年份必须是4位数字，例如 2026。")
    year = int(compact)
    if year < 1 or year > 9998:
        raise ValueError("年份超出支持范围。")
    return year


def _friday_week_start(day):
    return day - timedelta(days=(day.weekday() - 4) % 7)


def _week_starts_for_month(month_start):
    _, last_day = calendar.monthrange(month_start.year, month_start.month)
    first_friday_offset = (4 - month_start.weekday()) % 7
    first_friday = month_start + timedelta(days=first_friday_offset)
    result = []
    current = first_friday
    while current.month == month_start.month and current.day <= last_day:
        result.append(current)
        current += timedelta(days=7)
    return result


def _show_range(
    payload: dict[str, object],
    title: str,
    start_day,
    end_day,
) -> int:
    runs = _sum_daily_range(payload, "daily_runs", start_day, end_day)
    successes = _sum_daily_range(
        payload,
        "daily_player_entries",
        start_day,
        end_day,
    )
    failures = _sum_daily_range(
        payload,
        "daily_player_entry_failures",
        start_day,
        end_day,
    )
    visits = _sum_nested_range(
        payload,
        "daily_player_visits",
        start_day,
        end_day,
    )
    tasks = _sum_nested_range(
        payload,
        "daily_player_completed_tasks",
        start_day,
        end_day,
    )
    returned_failures = _sum_nested_range(
        payload,
        "daily_player_returned_before_arrival",
        start_day,
        end_day,
    )
    room_closed_failures = _sum_nested_range(
        payload,
        "daily_player_room_closed_before_arrival",
        start_day,
        end_day,
    )
    network_error_failures = _sum_nested_range(
        payload,
        "daily_player_network_error_before_arrival",
        start_day,
        end_day,
    )

    print(title)
    print(
        f"统计范围：北京时间 {start_day.isoformat()} 05:00:00 "
        f"至 {end_day.isoformat()} 04:59:59"
    )
    print(f"CODE更新轮次：{runs}")
    print(f"成功服务人次：{successes}")
    print(f"进入失败人次：{failures}")
    _show_daily_service_table(payload, start_day, end_day)
    _show_player_ranking(
        "玩家来访次数（按次数从高到低）：",
        visits,
        "到访",
    )
    _show_player_ranking(
        "玩家参与完成任务数（按次数从高到低）：",
        tasks,
        "参与完成任务",
    )
    _show_failure_ranking(
        "玩家进入失败次数（按次数从高到低）：",
        returned_failures,
        room_closed_failures,
        network_error_failures,
        failures,
    )
    return 0


def _show_week(payload: dict[str, object]) -> int:
    try:
        raw_month = input(
            "请输入年月（例如 202609，留空=本周）："
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消查询。")
        return 1
    if not raw_month:
        start_day = _friday_week_start(_current_business_day())
        return _show_range(
            payload,
            "本周统计",
            start_day,
            start_day + timedelta(days=7),
        )
    try:
        month_start = _parse_month(raw_month)
    except ValueError as exc:
        print(f"[错误] {exc}")
        return 2
    weeks = _week_starts_for_month(month_start)
    print()
    print("请选择周：")
    for index, start_day in enumerate(weeks, 1):
        end_day = start_day + timedelta(days=7)
        print(
            f"{index} = {start_day.strftime('%Y.%m.%d')} 05:00～"
            f"{end_day.strftime('%Y.%m.%d')} 04:59:59"
        )
    try:
        raw_selection = input("请输入周编号（留空=第1周）：").strip() or "1"
        selection = int(raw_selection)
    except (EOFError, KeyboardInterrupt):
        print("\n已取消查询。")
        return 1
    except ValueError:
        print("[错误] 周编号必须是数字。")
        return 2
    if selection < 1 or selection > len(weeks):
        print(f"[错误] 请输入1～{len(weeks)}。")
        return 2
    start_day = weeks[selection - 1]
    return _show_range(
        payload,
        f"{month_start.strftime('%Y年%m月')}第{selection}周统计",
        start_day,
        start_day + timedelta(days=7),
    )


def _show_month(payload: dict[str, object]) -> int:
    try:
        raw_month = input(
            "请输入年月（例如 202609，留空=本月）："
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消查询。")
        return 1
    try:
        month_start = (
            _parse_month(raw_month)
            if raw_month
            else _current_business_day().replace(day=1)
        )
    except ValueError as exc:
        print(f"[错误] {exc}")
        return 2
    if month_start.month == 12:
        next_month = month_start.replace(
            year=month_start.year + 1,
            month=1,
        )
    else:
        next_month = month_start.replace(month=month_start.month + 1)
    return _show_range(
        payload,
        month_start.strftime("%Y年%m月统计"),
        month_start,
        next_month,
    )


def _show_year(payload: dict[str, object]) -> int:
    try:
        raw_year = input("请输入年份（例如 2026，留空=本年）：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消查询。")
        return 1
    try:
        year = _parse_year(raw_year) if raw_year else _current_business_day().year
    except ValueError as exc:
        print(f"[错误] {exc}")
        return 2
    start_day = datetime(year, 1, 1).date()
    end_day = datetime(year + 1, 1, 1).date()
    return _show_range(payload, f"{year}年统计", start_day, end_day)


def _show_period_menu(payload: dict[str, object]) -> int:
    print("请选择统计周期：")
    print("1 = 日统计")
    print("2 = 周统计")
    print("3 = 月统计")
    print("4 = 年统计")
    try:
        selection = input("请输入 1～4（留空=日统计）：").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print("\n已取消查询。")
        return 1
    if selection == "1":
        print()
        return _show_one_day(payload)
    if selection == "2":
        print()
        return _show_week(payload)
    if selection == "3":
        print()
        return _show_month(payload)
    if selection == "4":
        print()
        return _show_year(payload)
    print("[错误] 请输入1、2、3、4，或直接回车。")
    return 2


def _show_one_day(payload: dict[str, object]) -> int:
    try:
        raw_day = input(
            "请输入要查询的日期（例如 20260904，留空=上一个业务日）："
        )
        day = _parse_business_day(raw_day) if raw_day.strip() else (
            _previous_business_day()
        )
    except (EOFError, KeyboardInterrupt):
        print("\n已取消查询。")
        return 1
    except ValueError as exc:
        print(f"[错误] {exc}")
        return 2

    successes = _counter_map(payload, "daily_player_entries")
    failures = _counter_map(payload, "daily_player_entry_failures")
    daily_visits = _nested_counter_map(payload, "daily_player_visits")
    daily_tasks = _nested_counter_map(
        payload,
        "daily_player_completed_tasks",
    )
    daily_returned_failures = _nested_counter_map(
        payload,
        "daily_player_returned_before_arrival",
    ).get(day, {})
    daily_room_closed_failures = _nested_counter_map(
        payload,
        "daily_player_room_closed_before_arrival",
    ).get(day, {})
    daily_network_error_failures = _nested_counter_map(
        payload,
        "daily_player_network_error_before_arrival",
    ).get(day, {})
    print()
    print(f"业务日期：{day}")
    print(f"统计范围：北京时间 {day} 05:00 至次日 04:59:59")
    print(f"成功服务人次：{successes.get(day, 0)}")
    print(f"进入失败人次：{failures.get(day, 0)}")
    _show_player_ranking(
        "玩家来访次数（当日，按次数从高到低）：",
        daily_visits.get(day, {}),
        "到访",
    )
    _show_player_ranking(
        "玩家参与完成任务数（当日，按次数从高到低）：",
        daily_tasks.get(day, {}),
        "参与完成任务",
    )
    _show_failure_ranking(
        "玩家进入失败次数（当日总计，按次数从高到低）：",
        daily_returned_failures,
        daily_room_closed_failures,
        daily_network_error_failures,
        failures.get(day, 0),
    )
    return 0


def _show_all_days(
    payload: dict[str, object],
    selection: str = "1",
) -> int:
    runs = _counter_map(payload, "daily_runs")
    successes = _counter_map(payload, "daily_player_entries")
    failures = _counter_map(payload, "daily_player_entry_failures")
    days = sorted(set(runs) | set(successes) | set(failures))

    if selection in {"1", "2"}:
        print("业务日期以北京时间早上 05:00 为分界。")
        print()
        if not days:
            print("目前没有每日玩家统计记录。")
        else:
            print(
                f"{'业务日期':<12} {'成功服务人次':>12} "
                f"{'进入失败人次':>12}"
            )
            print("-" * 42)
            for day in days:
                print(
                    f"{day:<12} {successes.get(day, 0):>12} "
                    f"{failures.get(day, 0):>12}"
                )

        print()
        print(
            "历史成功总人次："
            f"{max(0, int(payload.get('total_player_entries', 0)))}"
        )
        print(
            "历史失败总人次："
            f"{max(0, int(payload.get('total_player_entry_failures', 0)))}"
        )
    total_visits = _counter_map(payload, "total_player_visits")
    total_tasks = _counter_map(payload, "total_player_completed_tasks")
    total_returned_failures = _counter_map(
        payload,
        "total_player_returned_before_arrival",
    )
    total_room_closed_failures = _counter_map(
        payload,
        "total_player_room_closed_before_arrival",
    )
    total_network_error_failures = _counter_map(
        payload,
        "total_player_network_error_before_arrival",
    )
    if selection in {"1", "3"}:
        _show_player_ranking(
            "玩家来访次数（历史总计，按次数从高到低）：",
            total_visits,
            "到访",
        )
    if selection in {"1", "4"}:
        _show_player_ranking(
            "玩家参与完成任务数（历史总计，按次数从高到低）：",
            total_tasks,
            "参与完成任务",
        )
    if selection in {"1", "5"}:
        _show_failure_ranking(
            "玩家进入失败次数（历史总计，按次数从高到低）：",
            total_returned_failures,
            total_room_closed_failures,
            total_network_error_failures,
            int(payload.get("total_player_entry_failures", 0)),
        )
    return 0


def _show_menu(payload: dict[str, object]) -> int:
    print("请选择要展示的内容：")
    print("1 = 全部展示")
    print("2 = 只展示业务日期、服务人次、失败人次")
    print("3 = 玩家来访查询")
    print("4 = 玩家任务查询")
    print("5 = 玩家失败查询")
    try:
        selection = input("请输入 1～5（留空=2）：").strip() or "2"
    except (EOFError, KeyboardInterrupt):
        print("\n已取消查询。")
        return 1
    if selection not in {"1", "2", "3", "4", "5"}:
        print("[错误] 请输入1、2、3、4或5。")
        return 2
    print()
    return _show_all_days(payload, selection)


def main() -> int:
    parser = argparse.ArgumentParser(description="查看 Macro6 玩家服务统计")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--date", action="store_true", help="交互查询指定日期")
    mode.add_argument("--all", action="store_true", help="输出全部日期")
    mode.add_argument("--menu", action="store_true", help="交互选择输出内容")
    mode.add_argument("--week", action="store_true", help="查询周统计")
    mode.add_argument("--month", action="store_true", help="查询月统计")
    mode.add_argument("--year", action="store_true", help="查询年统计")
    mode.add_argument(
        "--period-menu",
        action="store_true",
        help="交互选择日、周、月、年统计",
    )
    args = parser.parse_args()

    try:
        payload = _load_stats()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[错误] {exc}")
        return 1

    if args.date:
        return _show_one_day(payload)
    if args.menu:
        return _show_menu(payload)
    if args.week:
        return _show_week(payload)
    if args.month:
        return _show_month(payload)
    if args.year:
        return _show_year(payload)
    if args.period_menu:
        return _show_period_menu(payload)
    return _show_all_days(payload)


if __name__ == "__main__":
    raise SystemExit(main())
