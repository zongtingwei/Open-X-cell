"""Build the legacy 4,000-gene HepG2 Nadig Mini validation subset."""
from __future__ import annotations

from prep_real_subset import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_output="data/real/hepg2nadig_mini4k.h5ad",
            default_panel=4_000,
        )
    )
