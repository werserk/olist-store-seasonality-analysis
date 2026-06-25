#!/usr/bin/env python3
"""Copy analysis figures into presentation/assets/figures for Slidev."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final_figure_set"
PRES = ROOT / "results" / "figures" / "pres"
OUT = ROOT / "presentation" / "assets" / "figures"

# final_figure_set -> presentation/assets/figures (same names)
EXTRA = {
    PRES / "08_detrended.png": OUT / "20_detrended_daily_orders_2017.png",
    PRES / "05_large_share_ci.png": OUT / "21_large_purchase_share_ci_2017.png",
    PRES / "06_weekday.png": OUT / "22_weekday_orders_2017.png",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if FINAL.is_dir():
        for src in sorted(FINAL.glob("*.png")):
            shutil.copy2(src, OUT / src.name)
    for src, dst in EXTRA.items():
        if src.exists():
            shutil.copy2(src, dst)
    print(f"OK {OUT} ({len(list(OUT.glob('*.png')))} png)")


if __name__ == "__main__":
    main()
