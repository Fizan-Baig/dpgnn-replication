# ---------------------------------------------------------------------------
# Runs one model with the OFFICIAL CRRS / RecBole-PJF code and saves its test metrics.
# Usage (from inside the cloned CRRS folder):
#   python run_official.py --model DPGNN --epochs 300 --out ../results/official_DPGNN.json
# The patches below only restore names removed by newer NumPy / pandas / PyTorch releases;
# they do not change any model, data or metric logic.
# ---------------------------------------------------------------------------
import argparse
import json
import os
import sys
import time

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# CRRS is built on RecBole 1.1.1, which uses np.float, np.bool, np.long and np.str (removed in NumPy 2);
# RecBole 1.2.0 additionally reads np.float_, np.complex_ and np.unicode_. All are restored here.
for alias, real in {"float_": np.float64, "complex_": np.complex128, "unicode_": np.str_, "string_": np.bytes_,
                    "NaN": np.nan, "Inf": np.inf, "float": float, "int": int, "bool": bool,
                    "object": object, "complex": complex, "long": int, "str": str}.items():
    try:
        getattr(np, alias)
    except AttributeError:
        setattr(np, alias, real)

import pandas as pd
if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda self, other, ignore_index=False, **kw: pd.concat(
        [self, other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])], ignore_index=ignore_index)

import scipy.sparse as sp
if not hasattr(sp.dok_matrix, "_update"):      # RecBole 1.1.1 LightGCN calls dok_matrix._update (removed in SciPy 1.13+)
    def _dok_update(self, d):
        if hasattr(self, "_dict"):
            self._dict.update(d)
        else:
            dict.update(self, d)
    sp.dok_matrix._update = _dok_update

import torch
_orig_load = torch.load
def _load(*args, **kwargs):          # RecBole checkpoints store config objects, not just weights
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _load

sys.path.insert(0, os.getcwd())
from recbole_pjf.quick_start import run_recbole_pjf


def to_float(v):
    try:
        return float(v.item()) if hasattr(v, "item") else float(v)
    except Exception:
        return str(v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DPGNN")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    res = run_recbole_pjf(model=args.model, dataset="libimseti",
                          config_dict={"epochs": args.epochs, "gpu_id": "0", "show_progress": False})
    side_a, side_b = res["test_result"]
    out = {"model": args.model, "epochs_cap": args.epochs, "seconds": time.time() - t0,
           "best_valid_score": to_float(res["best_valid_score"]),
           "side_A": {k: to_float(v) for k, v in side_a.items()},
           "side_B": {k: to_float(v) for k, v in side_b.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
