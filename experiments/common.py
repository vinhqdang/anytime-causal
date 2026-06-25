"""Shared utilities for experiments: IO, plotting, checkpoint harness."""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def save_json(name: str, obj) -> str:
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    return path


def load_json(name: str):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def figpath(name: str) -> str:
    return os.path.join(FIG_DIR, name)


def newfig(*a, **k):
    return plt.subplots(*a, **k)


def savefig(fig, name: str) -> str:
    p = figpath(name)
    fig.tight_layout()
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p
