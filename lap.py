"""
Minimal stub of the optional `lap` dependency.

Ultralytics' tracking utilities import `lap` for linear assignment. This app does not
use Ultralytics trackers (it uses detection + a lightweight tracker), but some
Ultralytics versions may still import `lap` at import-time in tracking modules.

Keeping this stub prevents a hard crash on platforms where `lap` wheels are not
available (e.g. Streamlit Cloud with newer Python versions).
"""

from __future__ import annotations

import numpy as np

__all__ = ["lapjv", "__version__"]

__version__ = "0.0.0-stub"


def lapjv(*args, **kwargs):
    """
    Very small fallback implementation of `lap.lapjv`.

    This is NOT a full Jonker–Volgenant solver. It uses a greedy assignment that is
    "good enough" to keep optional tracker codepaths from crashing when the real
    `lap` wheel isn't available.
    """
    if not args:
        raise TypeError("lapjv(cost_matrix, ...) missing required argument: 'cost_matrix'")

    cost_matrix = np.asarray(args[0], dtype=np.float32)
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2D")

    cost_limit = kwargs.get("cost_limit", None)
    if cost_limit is None:
        cost_limit = float("inf")
    cost_limit = float(cost_limit)

    n_rows, n_cols = cost_matrix.shape
    x = np.full((n_rows,), -1, dtype=np.int32)
    y = np.full((n_cols,), -1, dtype=np.int32)

    # Greedy: smallest costs first.
    pairs = []
    for i in range(n_rows):
        row = cost_matrix[i]
        for j in range(n_cols):
            c = float(row[j])
            if c <= cost_limit:
                pairs.append((c, i, j))
    pairs.sort(key=lambda t: t[0])

    used_rows = set()
    used_cols = set()
    total_cost = 0.0
    for c, i, j in pairs:
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        x[i] = j
        y[j] = i
        total_cost += c

    return total_cost, x, y
