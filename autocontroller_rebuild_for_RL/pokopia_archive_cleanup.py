#!/usr/bin/env python3
"""Delete non-error Pokopia images older than the configured retention period."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import time


DEFAULT_ROOT = Path(__file__).resolve().with_name("pokopia_stamp_records")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
TIMESTAMP_PATTERN = re.compile(r"(20\d{6})T(\d{6})(?:_\d+)?Z", re.IGNORECASE)


@dataclass(frozen=True)
class CleanupResult:
    scanned: int
    deleted: int
    freed_bytes: int
    failures: int


def _image_epoch(path: Path) -> float:
    match = TIMESTAMP_PATTERN.search(path.name)
    if match:
        try:
            parsed = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
            return parsed.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    return path.stat().st_mtime


def cleanup_archives(
    root: Path = DEFAULT_ROOT,
    *,
    retention_days: int = 30,
    dry_run: bool = False,
    now_epoch: float | None = None,
) -> CleanupResult:
    """Remove old non-ERROR images, while retaining JSON/text logs forever."""
    keep_days = max(1, int(retention_days))
    cutoff = (time.time() if now_epoch is None else float(now_epoch)) - keep_days * 86400
    scanned = deleted = freed_bytes = failures = 0
    if not root.is_dir():
        return CleanupResult(0, 0, 0, 0)

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        scanned += 1
        # OCR/network diagnostic evidence is explicitly permanent.
        if "ERROR" in path.name.upper():
            continue
        try:
            if _image_epoch(path) >= cutoff:
                continue
            size = path.stat().st_size
            if not dry_run:
                path.unlink()
            deleted += 1
            freed_bytes += size
        except OSError:
            failures += 1
    return CleanupResult(scanned, deleted, freed_bytes, failures)


def main() -> int:
    parser = argparse.ArgumentParser(description="清理一个月前的Pokopia非ERROR图片")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="只统计，不删除")
    args = parser.parse_args()
    result = cleanup_archives(args.root, retention_days=args.days, dry_run=args.dry_run)
    action = "预计删除" if args.dry_run else "已删除"
    print(
        f"扫描{result.scanned}张图片，{action}{result.deleted}张，"
        f"释放{result.freed_bytes / 1024 / 1024:.1f}MB，失败{result.failures}张。"
    )
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
