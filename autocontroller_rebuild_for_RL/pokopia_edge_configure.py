#!/usr/bin/env python3
"""Interactive Windows setup for the Pokopia edge uploader."""

from __future__ import annotations

import getpass
import sys

if __package__:
    from .pokopia_edge_config import CONFIG_PATH, save_edge_config
else:
    from pokopia_edge_config import CONFIG_PATH, save_edge_config


def main() -> int:
    if sys.platform != "win32":
        print("该配置工具需要在Windows上运行。")
        return 1
    print("Pokopia Cloudflare边缘上传配置")
    print("令牌会用Windows DPAPI加密，只能由当前Windows用户在本机解密。")
    base_url = input("公开地址 [https://stamp.rabi.date]：").strip() or "https://stamp.rabi.date"
    token = getpass.getpass("粘贴Cloudflare Worker的UPLOAD_TOKEN（输入不可见）：").strip()
    try:
        save_edge_config(base_url, token)
    except Exception as exc:
        print(f"配置失败：{exc}")
        return 1
    print(f"配置已保存：{CONFIG_PATH}")
    print("令牌没有以明文写入文件。重新运行本工具可替换配置。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
