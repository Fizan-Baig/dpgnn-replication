# ---------------------------------------------------------------------------
# Evaluation, mirroring RecBole (Recall, Precision, NDCG, MRR @K) and the
# reciprocal metrics added in CRRS's MultiDirectTrainer.evaluate():
#   CRecall    = |pairs found by side A OR side B| / |test match pairs|
#   CPrecision = |pairs found by A OR B| / (K * (users_A + users_B))
#   SRecall    = 2 * |pairs found by A AND B| / |test match pairs|
#   SPrecision = 2 * |pairs found by A AND B| / (K * (users_A + users_B))
#   RNDCG@K    = user-weighted average of the two sides' NDCG@K
# Side A = each man ranks all women; side B = each woman ranks all men.
# Only users with at least one target match are evaluated on a side.
# ---------------------------------------------------------------------------
import numpy as np


def mask_and_topk_numpy(scores, history, k):
    """Reference implementation: hide history pairs, return top-k column indices per row (best first)."""
    s = scores.astype(np.float64, copy=True)
    if len(history):
        s[history[:, 0], history[:, 1]] = -np.inf
    part = np.argpartition(-s, k, axis=1)[:, :k]
    rows = np.arange(s.shape[0])[:, None]
    order = np.argsort(-s[rows, part], axis=1, kind="stable")
    return part[rows, order]


def _side_metrics(topk, targets_by_user, k):
    """topk: [n_rows, k] ranked candidate ids; targets_by_user: dict row -> set of true ids."""
    users = np.array(sorted(targets_by_user.keys()))
    hits = np.zeros((len(users), k), dtype=bool)
    pos_len = np.zeros(len(users), dtype=np.int64)
    for i, u in enumerate(users):
        t = targets_by_user[u]
        pos_len[i] = len(t)
        hits[i] = np.fromiter((c in t for c in topk[u]), dtype=bool, count=k)
    n_hit = hits.sum(1)
    recall = n_hit / pos_len
    precision = n_hit / k
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc).sum(1)
    idcg = np.cumsum(disc)[np.minimum(pos_len, k) - 1]
    ndcg = dcg / idcg
    first = np.where(hits.any(1), hits.argmax(1) + 1, np.inf)
    mrr = np.where(np.isfinite(first), 1.0 / first, 0.0)
    found = [(u, c) for i, u in enumerate(users) for c, h in zip(topk[u], hits[i]) if h]
    return {"recall": recall.mean(), "precision": precision.mean(), "ndcg": ndcg.mean(),
            "mrr": mrr.mean(), "n_users": len(users)}, found


def evaluate_topk(topk_a, topk_b, targets, k=50):
    """topk_a: [n_men, k] women ids; topk_b: [n_women, k] men ids; targets: [n, 2] (man, woman) pairs."""
    by_m, by_f = {}, {}
    for m, f in targets:
        by_m.setdefault(int(m), set()).add(int(f))
        by_f.setdefault(int(f), set()).add(int(m))
    res_a, found_a = _side_metrics(topk_a, by_m, k)
    res_b, found_b = _side_metrics(topk_b, by_f, k)
    pairs_a = set(found_a)                              # (man, woman)
    pairs_b = {(m, f) for f, m in found_b}              # flip to (man, woman)
    total = len({(int(m), int(f)) for m, f in targets})
    n_users = res_a["n_users"] + res_b["n_users"]
    union, inter = len(pairs_a | pairs_b), len(pairs_a & pairs_b)
    out = {f"A_{key}@{k}" if key != "n_users" else "A_users": v for key, v in res_a.items()}
    out.update({f"B_{key}@{k}" if key != "n_users" else "B_users": v for key, v in res_b.items()})
    out.update({
        f"recall@{k}": (res_a["recall"] + res_b["recall"]) / 2,       # "average of both sides" (CRRS Table)
        f"ndcg@{k}": (res_a["ndcg"] + res_b["ndcg"]) / 2,
        f"precision@{k}": (res_a["precision"] + res_b["precision"]) / 2,
        f"crecall@{k}": union / total,
        f"cprecision@{k}": union / (k * n_users),
        f"srecall@{k}": 2 * inter / total,
        f"sprecision@{k}": 2 * inter / (k * n_users),
        f"rndcg@{k}": (res_a["ndcg"] * res_a["n_users"] + res_b["ndcg"] * res_b["n_users"]) / n_users,
        "true_positive_pairs": union,
        "target_pairs": total,
    })
    return {key: float(v) for key, v in out.items()}


def evaluate_scores_numpy(scores, data, phase, k=50, all_train=False):
    """Full-ranking evaluation from a dense [n_men, n_women] score matrix (reference path, CPU)."""
    hist = data.history(phase, all_train)
    topk_a = mask_and_topk_numpy(scores, hist, k)
    topk_b = mask_and_topk_numpy(scores.T, hist[:, ::-1], k)
    return evaluate_topk(topk_a, topk_b, data.targets(phase), k)
