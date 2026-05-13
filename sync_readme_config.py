"""Synchronize README default-configuration section from src/config.py.

Usage:
    python sync_readme_config.py
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import Config  # noqa: E402

AUTO_START = "<!-- AUTO-CONFIG-START -->"
AUTO_END = "<!-- AUTO-CONFIG-END -->"


def _fmt_value(value) -> str:
    if isinstance(value, str):
        return f"`{value}`"
    return f"`{value!r}`"


def _table_for_dataclass(title: str, obj) -> list[str]:
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass for {title}")

    out = [f"### {title}", "", "| Key | Value |", "|---|---|"]
    for f in fields(obj):
        out.append(f"| `{f.name}` | {_fmt_value(getattr(obj, f.name))} |")
    out.append("")
    return out


def _derived_problem_table(cfg: Config) -> list[str]:
    p = cfg.problem
    out = [
        "### ProblemConfig Derived Values",
        "",
        "| Key | Value |",
        "|---|---|",
        f"| `u_ref` | `{p.u_ref!r}` |",
        f"| `eta_max` | `{p.eta_max!r}` |",
        f"| `hole_xi_c` | `{p.hole_xi_c!r}` |",
        f"| `hole_eta_c` | `{p.hole_eta_c!r}` |",
        f"| `hole_rc` | `{p.hole_rc!r}` |",
        "",
    ]
    return out


def build_auto_block() -> str:
    cfg = Config()

    lines: list[str] = [
        AUTO_START,
        "_This section is auto-generated from `src/config.py` by `python sync_readme_config.py`._",
        "",
    ]
    lines.extend(_table_for_dataclass("ProblemConfig", cfg.problem))
    lines.extend(_derived_problem_table(cfg))
    lines.extend(_table_for_dataclass("NetworkConfig", cfg.network))
    lines.extend(_table_for_dataclass("TrainingConfig", cfg.training))
    lines.extend(_table_for_dataclass("PlotConfig", cfg.plotting))
    lines.append(AUTO_END)
    return "\n".join(lines)


def upsert_readme(readme_path: Path) -> None:
    text = readme_path.read_text(encoding="utf-8")
    block = build_auto_block()

    if AUTO_START in text and AUTO_END in text:
        start = text.index(AUTO_START)
        end = text.index(AUTO_END) + len(AUTO_END)
        new_text = text[:start] + block + text[end:]
    else:
        anchor = "## Current Default Setup"
        idx = text.find(anchor)
        if idx == -1:
            raise RuntimeError("Could not find '## Current Default Setup' in README.md")
        insert_at = idx + len(anchor)
        new_text = text[:insert_at] + "\n\n" + block + text[insert_at:]

    readme_path.write_text(new_text, encoding="utf-8")


if __name__ == "__main__":
    upsert_readme(ROOT / "README.md")
    print("README.md synchronized with src/config.py")
