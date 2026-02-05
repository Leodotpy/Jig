"""Console UI — colored output, prompts, and JSON pretty-printing."""

import json
import re
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from colorama import Fore, Style, init

init(autoreset=True, strip=False)

# Color shortcuts
C = Fore.CYAN
G = Fore.GREEN
Y = Fore.YELLOW
R = Fore.RED
M = Fore.MAGENTA
DIM = Style.DIM
BRIGHT = Style.BRIGHT
RESET = Style.RESET_ALL


def print_json_colored(data: Dict[str, Any], indent: int = 2) -> None:
    """Pretty print JSON with syntax highlighting."""
    text = json.dumps(data, indent=indent, ensure_ascii=False)
    for line in text.split("\n"):
        highlighted = line
        highlighted = re.sub(r'(".*?"): ', rf"{C}\1{RESET}: ", highlighted)
        highlighted = re.sub(r': (".*?")([,]?)', rf": {G}\1{RESET}\2", highlighted)
        highlighted = re.sub(r": ([\d.]+)([,]?)", rf": {Y}\1{RESET}\2", highlighted)
        highlighted = re.sub(
            r": (true|false|null)([,]?)", rf": {M}\1{RESET}\2", highlighted
        )
        print(highlighted)


def confirm(msg: str, default: bool = False) -> bool:
    """Colored yes/no prompt."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    print(f"{Y}{BRIGHT}?{RESET} {msg}{suffix}", end="")
    try:
        response = input().strip().lower()
        if not response:
            return default
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def make_confirm(skip_all: bool = False) -> Callable[[str], bool]:
    """Create a confirm callback. When skip_all (--yes flag), always returns True."""
    if skip_all:
        return lambda _: True
    return lambda msg: confirm(msg, default=False)


class ConsoleUI:
    """Console output helpers for CLI."""

    @staticmethod
    def banner(version: str = "1.01") -> None:
        """Display startup banner."""
        print(f"\n{BRIGHT}{C}  Jig v{version}{RESET}")
        print(f"{DIM}  Structured Output Pipeline{RESET}\n")

    @staticmethod
    def heading(title: str) -> None:
        """Highlight a section heading."""
        print(f"{BRIGHT}{M}{title}{RESET}")

    @staticmethod
    def label(title: str) -> None:
        """Label a following block of text."""
        print(f"{Y}{BRIGHT}{title}:{RESET}")

    @staticmethod
    def block(text: str) -> None:
        """Render multi-line text with a subtle prefix for readability."""
        content = (text or "").rstrip("\n")
        if not content.strip():
            print(f"{DIM}│ (empty){RESET}")
            return
        for line in content.splitlines():
            print(f"{DIM}│ {RESET}{line}")

    @staticmethod
    def command_token(token: str) -> str:
        """Return a colorized command/token for inline help text."""
        return f"{BRIGHT}{C}{token}{RESET}"

    @staticmethod
    def command_list(entries: Sequence[Tuple[str, str]]) -> None:
        """Render aligned command + description pairs."""
        if not entries:
            return
        width = max(len(name) for name, _ in entries)
        for name, desc in entries:
            formatted = f"{ConsoleUI.command_token(name.ljust(width))}  {desc}"
            print(formatted)

    @staticmethod
    def success(msg: str) -> None:
        print(f"{G}{BRIGHT}[OK]{RESET} {msg}")

    @staticmethod
    def error(msg: str) -> None:
        print(f"{R}{BRIGHT}Error{RESET} {msg}")

    @staticmethod
    def info(msg: str, dim: bool = False) -> None:
        style = DIM if dim else ""
        print(f"{style}{msg}{RESET}")

    @staticmethod
    def dim(msg: str) -> None:
        print(f"{DIM}{msg}{RESET}")
