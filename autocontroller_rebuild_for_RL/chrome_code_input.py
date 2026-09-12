"""Send a Pokopia CODE message through an open remote-debugging Chrome tab.

The public ``input_code_to_chrome`` interface fills the Xiaohongshu chat
textbox and presses Enter to send it.
"""

from __future__ import annotations

import argparse
import os
import re
import threading
from dataclasses import dataclass


DEFAULT_CDP_URL = "http://127.0.0.1:9222"
DEFAULT_PAGE_TITLE = "消息 - 小红书"
DEFAULT_TEXT_TEMPLATE = "{code}"
_CODE_PATTERN = re.compile(r"^[A-Z0-9]{6}$")
_CHROME_INPUT_LOCK = threading.Lock()


@dataclass(frozen=True)
class ChromeCodeInputResult:
    code: str
    text: str
    page_title: str


def _expanded_text(code: str) -> str:
    template = os.environ.get(
        "MACRO6_CHROME_TEXT_TEMPLATE",
        DEFAULT_TEXT_TEMPLATE,
    )
    if "{code}" not in template:
        raise ValueError(
            "MACRO6_CHROME_TEXT_TEMPLATE 必须包含 {code}。"
        )
    try:
        return template.format(code=code)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "MACRO6_CHROME_TEXT_TEMPLATE 必须是包含 {code} 的有效模板。"
        ) from exc


def input_code_to_chrome(
    code: str,
    *,
    expanded_text: str | None = None,
    send: bool = True,
) -> ChromeCodeInputResult:
    """Fill expanded CODE text and optionally press Enter to send it.

    Chrome must already be running with a remote-debugging port.  Configuration
    can be overridden with these environment variables:

    - ``MACRO6_CHROME_CDP_URL`` (default ``http://127.0.0.1:9222``)
    - ``MACRO6_CHROME_PAGE_TITLE`` (default ``消息 - 小红书``)
    - ``MACRO6_CHROME_TEXT_TEMPLATE`` (default ``{code}``)
    """
    normalized_code = str(code or "").strip().upper()
    if not _CODE_PATTERN.fullmatch(normalized_code):
        raise ValueError("Chrome 输入接口只接受六位数字或大写字母 CODE。")

    text = (
        _expanded_text(normalized_code)
        if expanded_text is None
        else str(expanded_text)
    )
    if normalized_code not in text.upper():
        raise ValueError("Chrome扩展文字中必须包含对应的六位CODE。")
    endpoint = os.environ.get("MACRO6_CHROME_CDP_URL", DEFAULT_CDP_URL)
    title_hint = os.environ.get(
        "MACRO6_CHROME_PAGE_TITLE",
        DEFAULT_PAGE_TITLE,
    )

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "缺少 Playwright；请在 Windows 执行：pip install playwright"
        ) from exc

    # Serialize calls so two fast CODE updates cannot write over one another.
    with _CHROME_INPUT_LOCK, sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(endpoint)
        pages = [
            page
            for context in browser.contexts
            for page in context.pages
        ]
        target_page = next(
            (page for page in pages if title_hint in page.title()),
            None,
        )
        if target_page is None:
            available_titles = [page.title() for page in pages]
            raise RuntimeError(
                f"未找到标题包含 {title_hint!r} 的 Chrome 页面；"
                f"当前页面：{available_titles!r}"
            )

        # Xiaohongshu currently uses a contenteditable chat composer.  The
        # textarea/input fallbacks keep this interface usable if its markup
        # changes while retaining the same accessible textbox behavior.
        selector_groups = (
            target_page.locator('[contenteditable="true"]:visible'),
            target_page.locator("textarea:visible"),
            target_page.locator(
                'input:visible:not([type="hidden"]):not([type="search"])'
            ),
        )
        textbox = None
        for candidates in selector_groups:
            if candidates.count() > 0:
                textbox = candidates.last
                break
        if textbox is None:
            raise RuntimeError("小红书消息页中没有找到可见文字输入框。")

        textbox.wait_for(state="visible", timeout=5000)
        lines = text.split("\n")
        textbox.fill(lines[0])
        for line in lines[1:]:
            # Xiaohongshu's contenteditable composer can swallow the first
            # newline from a multiline fill().  Insert every line break as an
            # explicit chat-editor Shift+Enter, then insert Unicode text.
            textbox.press("Shift+Enter")
            if line:
                target_page.keyboard.insert_text(line)
        if send:
            textbox.press("Enter")
        return ChromeCodeInputResult(
            code=normalized_code,
            text=text,
            page_title=target_page.title(),
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把六位 CODE 填入远程调试 Chrome并发送。"
    )
    parser.add_argument("code", help="六位数字/字母 CODE")
    parser.add_argument(
        "--text",
        help="要输入的完整测试文字；其中必须包含对应的六位CODE。",
    )
    parser.add_argument(
        "--no-send",
        action="store_true",
        help="只填入聊天框，不按Enter发送。",
    )
    args = parser.parse_args()
    result = input_code_to_chrome(
        args.code,
        expanded_text=args.text,
        send=not args.no_send,
    )
    action = "输入但未发送" if args.no_send else "输入并发送"
    print(
        f"已在 {result.page_title!r} {action} {result.text!r}。",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
