#!/usr/bin/env python3
"""CPU round-trip test for stat-prompt composite conditioning (no GPU / vLLM).

Validates that a composite `data` token (label|m..|s..|n..|x..) survives:
  prompt construction -> generated-text format -> parser -> label filter -> bin parse.
Run on the cluster in sdforger-py312 before the GPU SFT job.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fms_dgt.public.databuilders.time_series.utils import convert_texts_to_tabular_data

NUM_COLS = [f"value_{k}" for k in range(5)]


def requested_bins_from_composite(comp: str) -> dict:
    out = {}
    for part in str(comp).split("|")[1:]:
        out[part[0]] = int(part[1:])
    return out


def filter_label(df: pd.DataFrame, requested_label: str) -> pd.DataFrame:
    prefix = df["data"].astype(str).str.split("|").str[0].str.strip()
    return df.loc[prefix == requested_label]


def fake_generated_text(composite: str, values: list[float]) -> str:
    inp = ", ".join(f"{c} is [blank]" for c in NUM_COLS)
    tgt = " ".join(f"{v:.4f} [answer]" for v in values)
    return f"Condition: data is {composite} [sep] Input: {inp} [sep] Target: {tgt}"


def main() -> None:
    # minimal training-like frame so the parser knows value_* numerical, data categorical
    original = pd.DataFrame(
        {**{c: np.random.RandomState(0).randn(8) for c in NUM_COLS},
         "data": ["walking|m1|s2|n0|x3"] * 4 + ["running|m2|s1|n3|x0"] * 4}
    )

    walk_vals = [0.1234, -0.5, 1.25, -1.1, 0.0]
    run_vals = [2.0, -2.0, 0.3, 0.4, -0.7]
    texts = [
        fake_generated_text("walking|m1|s2|n0|x3", walk_vals),
        fake_generated_text("running|m2|s1|n3|x0", run_vals),
        "Condition: data is walking|m0|s0|n0|x0 [sep] GARBAGE malformed line",  # should be dropped
    ]

    parsed = convert_texts_to_tabular_data(
        text=texts, original_dataset=original, text_template="fim_template_textual_encoding")
    for c in NUM_COLS:
        parsed[c] = pd.to_numeric(parsed[c], errors="coerce")
    parsed = parsed.dropna(subset=NUM_COLS)

    assert parsed.shape[0] == 2, f"expected 2 well-formed rows, got {parsed.shape[0]}"

    walk = filter_label(parsed, "walking")
    run = filter_label(parsed, "running")
    assert len(walk) == 1 and len(run) == 1, f"label filter wrong: walk={len(walk)} run={len(run)}"

    # composite preserved
    assert walk.iloc[0]["data"] == "walking|m1|s2|n0|x3"
    # values round-trip
    got = [float(walk.iloc[0][c]) for c in NUM_COLS]
    assert np.allclose(got, walk_vals, atol=1e-3), f"value mismatch: {got} vs {walk_vals}"
    # bin parse
    bins = requested_bins_from_composite("walking|m1|s2|n0|x3")
    assert bins == {"m": 1, "s": 2, "n": 0, "x": 3}, bins

    print("ROUND-TRIP OK: 2 parsed, label filter ok, composite + values + bins round-trip correctly.")
    print(parsed[["data"] + NUM_COLS].to_string(index=False))


if __name__ == "__main__":
    main()
