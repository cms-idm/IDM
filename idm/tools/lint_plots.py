#!/usr/bin/env python3
"""Plot linter — mechanical checks for the CMS/mplhep plotting conventions.

Vendored and lightly adapted from the JFC framework (violatingcp/jfc, "slopspec";
companion to Moreno, Bright-Thonney, Novak, Garcia, Harris, "AI Agents Can Already
Autonomously Perform Experimental High Energy Physics"). Pure standard library.

Scans Python plotting scripts for mechanical violations and exits 1 if any are found.
Pairs with ``idm.tools.plotting`` (the helpers that follow these conventions).

Usage:
    python -m idm.tools.lint_plots [dir]   # default: current directory
"""

import re
import sys
from pathlib import Path

_SKIP = ("__pycache__", "/.git/", "_venv/", "/idm_venv/", "/build/")


_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+(?:matplotlib|mplhep)\b", re.M)


def find_plotting_scripts(root: Path) -> list:
    """Find Python files that actually plot (import matplotlib or mplhep).

    Matches real import statements, not string mentions, and skips this linter itself —
    so a file that only names these libraries in comments/strings is not linted.
    """
    scripts = []
    for p in sorted(root.rglob("*.py")):
        sp = str(p)
        if p.name == "lint_plots.py" or any(s in sp for s in _SKIP):
            continue
        try:
            head = p.read_text(errors="replace")
        except Exception:
            continue
        if _IMPORT_RE.search(head):
            scripts.append(p)
    return scripts


def lint_file(path: Path) -> list:
    """Return a list of violation strings for a single file."""
    violations = []
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    def v(lineno, msg):
        violations.append(f"VIOLATION: {path}:{lineno} — {msg}")

    # --- Banned patterns ---
    banned = [
        (r"ax\.set_title\(", "No ax.set_title() — titles/captions go in the note, not the plot"),
        (r"fontsize\s*=\s*\d", "No absolute fontsize=N — use the stylesheet or 'x-small'"),
        (r"plt\.colorbar\(", "No plt.colorbar() — use make_square_add_cbar / cbarextend=True"),
        (r"fig\.colorbar\([^)]*\bax\s*=", "No fig.colorbar(ax=) — use fig.colorbar(cax=cax)"),
        (r"ax\.step\(", "No ax.step() for histograms — use mplhep histplot()"),
        (r"ax\.bar\(", "No ax.bar() for histograms — use mplhep histplot()"),
        (r"ax\.text\(", "No ax.text() — use mplhep label.add_text()"),
        (r"ax\.annotate\(", "No ax.annotate() — use mplhep label.add_text()"),
        (r"tight_layout\(\)", "No tight_layout() — use bbox_inches='tight' in savefig"),
    ]
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("#"):
            continue
        for pattern, msg in banned:
            if re.search(pattern, line):
                v(i, msg)

    # --- Derived-quantity error-bar trap: histtype="errorbar" on a .view() assignment w/o yerr ---
    view_lines = [i for i, line in enumerate(lines, 1) if re.search(r"\.view\(\)\s*\[.*\]\s*[+]?=", line)]
    if view_lines:
        for i, line in enumerate(lines, 1):
            if 'histtype="errorbar"' in line or "histtype='errorbar'" in line:
                context = "\n".join(lines[max(0, i - 6): i + 4])
                if "yerr=" not in context:
                    v(i, f'histtype="errorbar" on a derived quantity without yerr= (view assignment '
                         f'at line(s) {view_lines}) — mplhep applies sqrt(N); pass yerr= explicitly')

    # --- data=False stacking trap ---
    for i, line in enumerate(lines, 1):
        if "data=False" in line and "llabel" in line:
            v(i, 'data=False with llabel= stacks "Simulation" — use data=True')

    # --- ratio panel: sharex without hspace=0 ---
    has_sharex = any("sharex=True" in line for line in lines)
    if has_sharex and not any(("hspace=0" in line or "hspace = 0" in line) for line in lines):
        v(0, "sharex=True without hspace=0 — the ratio panel will have a gap")

    # --- single-panel figure size must be (10, 10) ---
    for i, line in enumerate(lines, 1):
        m = re.search(r"figsize\s*=\s*\((\d+),\s*(\d+)\)", line)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            if w == h and w != 10:
                v(i, f"figsize=({w}, {h}) — single-panel figures should be (10, 10)")

    # --- save format: both PDF and PNG, with bbox_inches + dpi ---
    has_savefig = any("savefig" in line for line in lines)
    if has_savefig:
        if not any(".pdf" in line and "savefig" in line for line in lines):
            v(0, "Missing PDF save — save both PDF and PNG")
        if not any(".png" in line and "savefig" in line for line in lines):
            v(0, "Missing PNG save — save both PDF and PNG")
        for i, line in enumerate(lines, 1):
            if "savefig(" in line:
                nxt = lines[i] if i < len(lines) else ""
                if "bbox_inches" not in line and "bbox_inches" not in nxt:
                    v(i, "savefig without bbox_inches='tight'")
                if "dpi=" not in line and "dpi=" not in nxt:
                    v(i, "savefig without dpi=200")

    # --- exp_label on ratio panels ---
    if sum(1 for line in lines if "exp_label(" in line) > 1 and has_sharex:
        v(0, "Multiple exp_label() calls on a sharex figure — label the main panel only")

    # --- bare underscores in labels (LaTeX) ---
    for i, line in enumerate(lines, 1):
        for fn in ("set_xlabel", "set_ylabel", "label="):
            if fn in line:
                m = re.search(r'["\']([^"\']+)["\']', line[line.index(fn):])
                if m:
                    outside_math = re.sub(r"\$[^$]+\$", "", m.group(1))
                    if "_" in outside_math and "\\_" not in outside_math:
                        v(i, f"Bare underscore in label '{m.group(1)}' — use LaTeX math ($..$) or escape")

    return violations


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    scripts = find_plotting_scripts(root)
    if not scripts:
        print(f"No plotting scripts (importing matplotlib/mplhep) found under {root}")
        return 0
    all_violations = []
    for s in scripts:
        all_violations.extend(lint_file(s))
    if all_violations:
        for line in all_violations:
            print(line)
        print(f"\n{len(all_violations)} violation(s) in {len(scripts)} file(s).")
        return 1
    print(f"No plotting violations in {len(scripts)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
