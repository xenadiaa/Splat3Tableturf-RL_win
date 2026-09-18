#!/usr/bin/env python3
"""Owner-only Cloudflare administration for community room data."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__:
    from .pokopia_edge_config import load_edge_config
else:
    from pokopia_edge_config import load_edge_config


ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = ROOT / "pokopia_community_backups"


def _request(path: str, *, method: str = "GET", payload=None):
    config = load_edge_config()
    body = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {config.upload_token}",
    }
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(
        config.base_url.rstrip("/") + path,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8-sig"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cloudflare返回HTTP {exc.code}：{detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接Cloudflare：{exc.reason}") from exc


def toggle() -> int:
    result = _request(
        "/api/admin/community/toggle",
        method="POST",
        payload={},
    )
    enabled = bool(result.get("writes_enabled"))
    print("玩家开门信息提交与反馈：" + ("已开启" if enabled else "已关闭"))
    print("发布者删除自己误填信息的权限不受此开关影响。")
    return 0


def export_backup(path: Path | None) -> int:
    result = _request("/api/admin/community/export")
    target = path
    if target is None:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        target = BACKUP_DIR / f"community_{datetime.now():%Y%m%d_%H%M%S}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"社区数据备份已下载：{target.resolve()}")
    return 0


def import_backup(path: Path, confirmation: str) -> int:
    if confirmation != "YES":
        raise RuntimeError("上传备份必须额外传入 --confirm-import YES。")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    result = _request(
        "/api/admin/community/import",
        method="POST",
        payload=payload,
    )
    print(
        "社区数据备份已合并上传："
        f"房间{result.get('rooms', 0)}条、反馈{result.get('feedback', 0)}条、"
        f"用量{result.get('usage', 0)}条。"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pokopia社区开门信息管理员接口")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("toggle", help="切换玩家提交与反馈总开关")
    export_parser = subparsers.add_parser("export", help="下载社区数据备份")
    export_parser.add_argument("path", nargs="?", type=Path)
    import_parser = subparsers.add_parser("import", help="合并上传社区数据备份")
    import_parser.add_argument("path", type=Path)
    import_parser.add_argument("--confirm-import", default="")
    args = parser.parse_args()
    if args.command == "toggle":
        return toggle()
    if args.command == "export":
        return export_backup(args.path)
    return import_backup(args.path, args.confirm_import)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"操作失败：{exc}")
        raise SystemExit(1)
