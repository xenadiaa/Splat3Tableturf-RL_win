#!/usr/bin/env python3
"""Guided, repeatable Cloudflare edge deployment for Pokopia."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
APP_DIR = ROOT / "autocontroller_rebuild_for_RL"
TEMPLATE = HERE / "wrangler.example.jsonc"
CONFIG = HERE / "wrangler.jsonc"
DATABASE_NAME = "pokopia-stamp"
BUCKET_NAME = "pokopia-stamp-media"
PUBLIC_URL = "https://stamp.rabi.date"
ERROR_LOG = ROOT / "pokopia_cloudflare_deploy_error.log"


def executable(name: str) -> str:
    candidate = f"{name}.cmd" if sys.platform == "win32" else name
    found = shutil.which(candidate)
    if not found:
        raise RuntimeError(
            f"未找到 {candidate}。请先安装Node.js LTS，关闭并重新打开CMD后再运行。"
        )
    return found


def cli(tool: str, *arguments: str) -> list[str]:
    """Run npm/npx through node.exe on Windows, bypassing cmd.exe quoting."""
    if sys.platform == "win32" and Path(tool).suffix.lower() in {".cmd", ".bat"}:
        tool_path = Path(tool).resolve()
        cli_name = f"{tool_path.stem.lower()}-cli.js"
        node_exe = tool_path.parent / "node.exe"
        cli_script = tool_path.parent / "node_modules" / "npm" / "bin" / cli_name
        if node_exe.is_file() and cli_script.is_file():
            return [str(node_exe), str(cli_script), *arguments]
        # Non-standard Node distributions may not place npm beside node.exe.
        # CALL delegates .cmd parsing to cmd without the broken /S quote removal.
        command_line = "call " + subprocess.list2cmdline([str(tool_path), *arguments])
        return [os.environ.get("ComSpec", "cmd.exe"), "/d", "/c", command_line]
    return [tool, *arguments]


def run(command: list[str], *, input_text: str | None = None, capture: bool = False, check: bool = True):
    print("\n> " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=HERE,
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    if check and result.returncode != 0:
        if capture and result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result


def parse_json_output(output: str):
    decoder = json.JSONDecoder()
    for start in (match.start() for match in re.finditer(r"[\[{]", output)):
        try:
            value, _ = decoder.raw_decode(output[start:])
            return value
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Wrangler返回内容中没有可读取的JSON。")


def d1_database_id(npx: str) -> str | None:
    result = run(cli(npx, "wrangler", "d1", "list", "--json"), capture=True)
    rows = parse_json_output(result.stdout)
    if isinstance(rows, dict):
        rows = rows.get("result", rows.get("databases", []))
    for row in rows if isinstance(rows, list) else []:
        if row.get("name") == DATABASE_NAME or row.get("database_name") == DATABASE_NAME:
            value = row.get("uuid") or row.get("id") or row.get("database_id")
            if value:
                return str(value)
    return None


def ensure_d1(npx: str) -> str:
    database_id = d1_database_id(npx)
    if database_id:
        print(f"已找到D1数据库：{DATABASE_NAME}")
        return database_id
    run(cli(npx, "wrangler", "d1", "create", DATABASE_NAME))
    database_id = d1_database_id(npx)
    if not database_id:
        raise RuntimeError("D1已创建但未能读取database_id，请稍后重新运行本脚本。")
    return database_id


def ensure_r2(npx: str) -> None:
    # Wrangler 4 supports --json for D1 list, but not for R2 bucket list.
    # Passing the unsupported flag can abort Node on some Windows builds.
    result = run(
        cli(npx, "wrangler", "r2", "bucket", "list"),
        capture=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode != 0:
        output = result.stdout or ""
        if "code: 10042" in output or "enable R2 through the Cloudflare Dashboard" in output:
            raise RuntimeError(
                "Cloudflare账户尚未启用R2。请在控制台进入 Storage & databases > "
                "R2 > Overview，完成R2订阅启用后重新运行本脚本；无需手工创建存储桶或API Token。"
            )
        raise subprocess.CalledProcessError(result.returncode, result.args, output=output)
    if BUCKET_NAME not in (result.stdout or ""):
        run(cli(npx, "wrangler", "r2", "bucket", "create", BUCKET_NAME))
    else:
        print(f"已找到R2存储桶：{BUCKET_NAME}")


def deploy_worker(npx: str) -> None:
    result = run(cli(npx, "wrangler", "deploy"), capture=True, check=False)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode == 0:
        return
    output = result.stdout or ""
    if "code: 10063" in output or "need a workers.dev subdomain" in output:
        raise RuntimeError(
            "Cloudflare账户尚未初始化Workers。请在Dashboard账户首页打开 Workers & Pages；"
            "如要求设置workers.dev账户子域，任选一个未占用名称确认即可。完成后重新运行本脚本，"
            "无需创建示例Worker；正式站仍只使用rabi.date和stamp.rabi.date。"
        )
    raise subprocess.CalledProcessError(result.returncode, result.args, output=output)


def write_wrangler(database_id: str) -> None:
    content = TEMPLATE.read_text(encoding="utf-8")
    content = content.replace("REPLACE_WITH_D1_DATABASE_ID", database_id)
    CONFIG.write_text(content, encoding="utf-8")
    print(f"已生成本机部署配置：{CONFIG.name}")


def existing_or_new_token() -> str:
    sys.path.insert(0, str(APP_DIR))
    from pokopia_edge_config import CONFIG_PATH, load_edge_config

    if CONFIG_PATH.is_file():
        try:
            return load_edge_config().upload_token
        except Exception:
            pass
    return secrets.token_urlsafe(48)


def save_windows_config(token: str) -> None:
    sys.path.insert(0, str(APP_DIR))
    from pokopia_edge_config import save_edge_config

    save_edge_config(PUBLIC_URL, token)


def main() -> int:
    if sys.platform != "win32":
        raise RuntimeError("此傻瓜式部署入口应在运行Macro6的Windows电脑上执行。")
    npm = executable("npm")
    npx = executable("npx")
    print("Pokopia Cloudflare边缘部署：不会开放路由器端口，也不会上传私密原始日志。")
    run(cli(npm, "install"))
    whoami = run(cli(npx, "wrangler", "whoami"), capture=True, check=False)
    if whoami.stdout:
        print(whoami.stdout, end="" if whoami.stdout.endswith("\n") else "\n")
    unauthenticated = "not authenticated" in (whoami.stdout or "").lower()
    if whoami.returncode != 0 or unauthenticated:
        print("即将打开浏览器，请登录托管 rabi.date 的Cloudflare账户并授权Wrangler。")
        run(cli(npx, "wrangler", "login"))

    database_id = ensure_d1(npx)
    ensure_r2(npx)
    write_wrangler(database_id)
    run(cli(npx, "wrangler", "d1", "migrations", "apply", "DB", "--remote"))
    deploy_worker(npx)

    token = existing_or_new_token()
    run(cli(npx, "wrangler", "secret", "put", "UPLOAD_TOKEN"), input_text=token + "\n")
    save_windows_config(token)
    print("\n部署完成。关闭当前窗口后，双击 run_pokopia_web_and_watchdog.bat 即会主动推送到边缘。")
    print(f"公开地址：{PUBLIC_URL}")
    print("如果自定义域部署失败，请先确认rabi.date已添加到同一Cloudflare账户且名称服务器状态为Active。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已取消。")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n部署失败：{exc}")
        ERROR_LOG.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"完整错误已保存：{ERROR_LOG}")
        raise SystemExit(1)
