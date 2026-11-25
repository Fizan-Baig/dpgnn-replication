# ---------------------------------------------------------------------------
# Results tables and figures for the replication.
# ---------------------------------------------------------------------------
import json
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Published Libimseti (dating) results, CRRS paper (Yang et al., KDD 2024), average of both sides, top 50.
PUBLISHED = {
    "BPRMF":    {"recall@50": 0.1552, "ndcg@50": 0.0661, "crecall@50": 0.2795},
    "LightGCN": {"recall@50": 0.1510, "ndcg@50": 0.0634, "crecall@50": 0.2549},
    "LFRR":     {"recall@50": 0.1729, "ndcg@50": 0.0706, "crecall@50": 0.3045},
    "DPGNN":    {"recall@50": 0.1825, "ndcg@50": 0.0739, "crecall@50": 0.3007},
}
METRICS = ["recall@50", "ndcg@50", "crecall@50"]
LABELS = {"recall@50": "Recall@50", "ndcg@50": "NDCG@50", "crecall@50": "CRecall@50",
          "srecall@50": "SRecall@50", "rndcg@50": "RNDCG@50"}
# names used by the official code for the same models
OFFICIAL_NAME = {"BPR": "BPRMF", "LightGCN": "LightGCN", "LFRR": "LFRR", "DPGNN": "DPGNN"}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_PUB, C_OFF, C_OURS = "#a3a29c", "#2a78d6", "#eb6834"   # published = neutral reference
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def style():
    plt.rcParams.update({
        "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": "white",
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left", "legend.frameon": False,
    })


def official_to_avg(rec):
    """Official JSON (side_A / side_B dicts) -> averaged metrics with our key names."""
    a, b = rec["side_A"], rec["side_B"]
    out = {}
    for key in ("recall@50", "ndcg@50", "precision@50", "mrr@50"):
        if key in a and key in b:
            out[key] = (a[key] + b[key]) / 2
            out["A_" + key], out["B_" + key] = a[key], b[key]
    for src, dst in (("crecall", "crecall@50"), ("srecall", "srecall@50"), ("rndcg@50", "rndcg@50"),
                     ("cprecision", "cprecision@50"), ("sprecision", "sprecision@50")):
        if src in a:
            out[dst] = a[src]
    return out


def ours_summary(runs):
    """runs: list of {'label', 'test': {...}} -> DataFrame of mean and std per label."""
    rows = []
    for r in runs:
        rows.append({"label": r["label"], **{k: v for k, v in r["test"].items() if isinstance(v, float)}})
    df = pd.DataFrame(rows)
    g = df.groupby("label", sort=False)
    return g.mean(numeric_only=True), g.std(numeric_only=True).fillna(0.0), g.size()


def comparison_table(ours_mean, ours_std, official):
    """One row per model: published, official-code run, our re-implementation (mean ± std)."""
    rows = []
    for model in ["DPGNN", "LFRR", "LightGCN", "BPRMF", "Popularity"]:
        row = {"Model": model}
        for m in METRICS:
            pub = PUBLISHED.get(model, {}).get(m)
            off = official.get(model, {}).get(m)
            row[f"{LABELS[m]} published"] = f"{pub:.4f}" if pub is not None else "–"
            row[f"{LABELS[m]} official code"] = f"{off:.4f}" if off is not None else "–"
            if model in ours_mean.index:
                mu, sd = ours_mean.loc[model, m], ours_std.loc[model, m]
                row[f"{LABELS[m]} ours"] = f"{mu:.4f} ± {sd:.4f}" if sd > 0 else f"{mu:.4f}"
            else:
                row[f"{LABELS[m]} ours"] = "–"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def fig_comparison(ours_mean, ours_std, official, path):
    """Grouped bars: published vs official code vs ours, for the models all three cover."""
    style()
    models = [m for m in ["BPRMF", "LightGCN", "LFRR", "DPGNN"] if m in PUBLISHED]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    w = 0.26
    x = np.arange(len(models))
    for ax, m in zip(axes, METRICS):
        pub = [PUBLISHED[k][m] for k in models]
        off = [official.get(k, {}).get(m, np.nan) for k in models]
        mu = [ours_mean.loc[k, m] if k in ours_mean.index else np.nan for k in models]
        sd = [ours_std.loc[k, m] if k in ours_std.index else 0 for k in models]
        ax.bar(x - w, pub, w - 0.02, color=C_PUB, label="Published (CRRS paper)")
        ax.bar(x, off, w - 0.02, color=C_OFF, label="Official code, our run")
        ax.bar(x + w, mu, w - 0.02, yerr=sd, color=C_OURS, label="Our re-implementation",
               error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
        ax.set_xticks(x, models); ax.set_title(LABELS[m]); ax.grid(axis="x", visible=False)
    fig.suptitle("Libimseti test set: published vs reproduced results (higher is better)",
                 x=0.01, y=0.99, ha="left", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=3, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(path, dpi=160); plt.close(fig)


def fig_curves(histories, path, metric="valid_crecall@50"):
    """Validation CRecall@50 and training loss per epoch, one line per model (first seed)."""
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    for (label, h), c in zip(histories.items(), SERIES):
        ep = [r["epoch"] for r in h]
        a1.plot(ep, [r[metric] for r in h], color=c, lw=2, label=label)
        best = int(np.argmax([r[metric] for r in h]))
        a1.plot(ep[best], h[best][metric], "o", color=c, ms=8, mec="white", mew=1.5)
        a2.plot(ep, [r["loss"] for r in h], color=c, lw=2, label=label)
    a1.set_xlabel("Epoch"); a1.set_ylabel("Validation CRecall@50"); a1.legend()
    a1.set_title("Validation CRecall@50 (dot = best epoch, kept for testing)")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Training loss"); a2.set_yscale("log"); a2.legend()
    a2.set_title("Training loss (log scale)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_bars(values, errors, title, ylabel, path, highlight=None):
    """Horizontal bars for one metric across variants (ablation / analysis)."""
    style()
    labels = list(values.keys())
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(labels) + 1.4))
    y = np.arange(len(labels))[::-1]
    colors = [C_OURS if l == highlight else C_OFF for l in labels]
    ax.barh(y, list(values.values()), xerr=[errors.get(l, 0) for l in labels], color=colors, height=0.6,
            error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
    for yi, l in zip(y, labels):
        ax.text(values[l] + errors.get(l, 0) + max(values.values()) * 0.01, yi, f"{values[l]:.4f}",
                va="center", color=INK, fontsize=10)
    ax.set_yticks(y, labels); ax.set_xlabel(ylabel); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values.values()) * 1.18); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_sides(ours_mean, path):
    """Side A (men ranking women) vs side B (women ranking men) Recall@50."""
    style()
    models = list(ours_mean.index)
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, ours_mean["A_recall@50"], w - 0.03, color=SERIES[0], label="Side A: men ranking women")
    ax.bar(x + w / 2, ours_mean["B_recall@50"], w - 0.03, color=SERIES[1], label="Side B: women ranking men")
    ax.set_xticks(x, models, rotation=15, ha="right"); ax.set_ylabel("Recall@50")
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=10)
    ax.grid(axis="x", visible=False); ax.set_title("Recall@50 for each side of the match (our re-implementation)", pad=30)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
