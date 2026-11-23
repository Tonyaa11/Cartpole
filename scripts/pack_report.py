"""
pack_report.py  —  Bundle a workshop report into a self-contained zip for Overleaf.

Usage
-----
    python scripts/pack_report.py 1          # workshop 1
    python scripts/pack_report.py 2          # workshop 2
    python scripts/pack_report.py 3          # workshop 3
    python scripts/pack_report.py 4          # workshop 4
    python scripts/pack_report.py all        # all workshops

Output
------
    dist/group5_workshop<N>_overleaf.zip

The zip contains a flat structure Overleaf expects:
    group5_workshop<N>.tex
    figures/<figure>.png   (from runs/workshop<N>/figures/)
    generated/<table>.tex  (from runs/workshop<N>/generated/, only W3/W4)

The .tex file inside the zip uses  figures/#3.png  and  generated/xxx  paths,
which is the Overleaf convention.  The source .tex in reports/ uses
\\graphicspath so that it compiles locally without copying anything.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"
DIST = ROOT / "dist"

WORKSHOPS = [1, 2, 3, 4]


def _tex_for_overleaf(src: Path, workshop: int) -> str:
    """Return tex source with paths reverted to Overleaf-relative form."""
    text = src.read_text(encoding="utf-8")

    # Point \graphicspath at figures/ only. The local line has nested braces,
    # \graphicspath{{../runs/workshopN/figures/}{figures/}}, so the pattern must
    # consume every {...} group (a plain [^}]+ stops at the first brace and leaves
    # a dangling "{figures/}}" that breaks the document).
    text = re.sub(r"\\graphicspath\{(?:\{[^{}]*\})+\}", lambda m: r"\graphicspath{{figures/}}", text)

    # Revert figure includes: #3.png -> figures/#3.png  (inside figph / figw)
    # \IfFileExists{X.png} -> \IfFileExists{figures/X.png}
    text = re.sub(
        r"\\IfFileExists\{([^/\}]+\.png)\}",
        r"\\IfFileExists{figures/\1}",
        text,
    )
    # \includegraphics[...]{X.png} -> \includegraphics[...]{figures/X.png}
    text = re.sub(
        r"\\includegraphics(\[[^\]]*\])\{([^/\}]+\.png)\}",
        r"\\includegraphics\1{figures/\2}",
        text,
    )
    # Revert \input paths: ../runs/workshopN/generated/X -> generated/X
    text = re.sub(
        r"\.\./runs/workshop\d+/generated/",
        "generated/",
        text,
    )
    return text


def pack(workshop: int) -> Path:
    name = f"group5_workshop{workshop}"
    tex_src = REPORTS / f"{name}.tex"
    fig_dir = RUNS / f"workshop{workshop}" / "figures"
    gen_dir = RUNS / f"workshop{workshop}" / "generated"
    out_zip = DIST / f"{name}_overleaf.zip"

    if not tex_src.exists():
        print(f"  SKIP  {tex_src} not found")
        return out_zip

    DIST.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # .tex with Overleaf-relative paths
        zf.writestr(f"{name}.tex", _tex_for_overleaf(tex_src, workshop))

        # figures
        if fig_dir.exists():
            for png in sorted(fig_dir.glob("*.png")):
                zf.write(png, f"figures/{png.name}")
        else:
            print(f"  WARN  no figures found at {fig_dir.relative_to(ROOT)}")

        # generated tables (W3 / W4 only)
        if gen_dir.exists():
            for f in sorted(gen_dir.iterdir()):
                zf.write(f, f"generated/{f.name}")

    print(f"  packed  {out_zip.relative_to(ROOT)}  ({out_zip.stat().st_size // 1024} KB)")
    return out_zip


def main() -> None:
    p = argparse.ArgumentParser(description="Pack workshop report for Overleaf")
    p.add_argument("workshop", help="workshop number (1-4) or 'all'")
    args = p.parse_args()

    if args.workshop.lower() == "all":
        targets = WORKSHOPS
    else:
        try:
            n = int(args.workshop)
            if n not in WORKSHOPS:
                raise ValueError
            targets = [n]
        except ValueError:
            print(f"error: invalid workshop '{args.workshop}' — must be 1, 2, 3, 4, or 'all'")
            sys.exit(1)

    for n in targets:
        pack(n)


if __name__ == "__main__":
    main()
