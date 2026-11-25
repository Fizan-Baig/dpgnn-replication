# ---------------------------------------------------------------------------
# Training loop with the official settings from CRRS (overall.yaml):
#   Adam, lr 1e-3, batch 4096, evaluate every epoch on validation,
#   early stopping on validation CRecall@50 with patience 30, test with the best checkpoint.
# ---------------------------------------------------------------------------
import copy
import random
import time

import numpy as np
import torch

from rep_data import sample_negatives
from rep_eval import evaluate_topk
from rep_models import DPGNN, LightGCN, BPRMF


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def topk_torch(scores, history, k):
    """Same as rep_eval.mask_and_topk_numpy, on the GPU."""
    s = scores.clone()
    if len(history):
        h = torch.as_tensor(history, dtype=torch.long, device=s.device)
        s[h[:, 0], h[:, 1]] = -float("inf")
    return torch.topk(s, k, dim=1).indices.cpu().numpy()


def evaluate_model(model, data, phase, k=50, mode="both", all_train=False):
    model.eval()
    scores = model.full_scores(mode)
    hist = data.history(phase, all_train)
    topk_a = topk_torch(scores, hist, k)
    topk_b = topk_torch(scores.T.contiguous(), hist[:, ::-1].copy(), k)
    model.train()
    return evaluate_topk(topk_a, topk_b, data.targets(phase), k)


MODELS = {"DPGNN": DPGNN, "LightGCN": LightGCN, "BPRMF": BPRMF}


def build_model(name, data, device, **kw):
    if name == "LightGCN":
        kw.setdefault("train_pairs", data.train_all)
    return MODELS[name](data, device, **kw).to(device)


def train(name, data, device, seed=2020, max_epochs=500, patience=30, batch_size=4096, lr=1e-3,
          k=50, valid_metric="crecall@50", verbose=True, **model_kw):
    """Train one model; returns (best model, history list, test metrics, info dict)."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(name, data, device, **model_kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # LightGCN in CRRS trains on every training interaction (REMAIN_NEG: True); the others on matches only
    pairs = data.train_all if name == "LightGCN" else data.train_pos
    if name == "LightGCN":
        neg_sets_m = [set() for _ in range(data.n_m)]; neg_sets_f = [set() for _ in range(data.n_f)]
        for m, f in pairs:
            neg_sets_m[m].add(f); neg_sets_f[f].add(m)
        sampler_data = type("S", (), {"n_m": data.n_m, "n_f": data.n_f, "m2f_train": neg_sets_m, "f2m_train": neg_sets_f})
    else:
        sampler_data = data
    # LightGCN is trained on every training interaction, so (as in the official code) all of them are hidden when ranking
    all_train = name == "LightGCN"
    best, best_state, best_epoch, wait, history = -1.0, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        perm = rng.permutation(len(pairs))
        total, n_batches = 0.0, 0
        for start in range(0, len(perm), batch_size):
            b = pairs[perm[start:start + batch_size]]
            neg_m, neg_f = sample_negatives(sampler_data, b[:, 0], b[:, 1], rng)
            t = lambda x: torch.as_tensor(x, dtype=torch.long, device=device)
            loss = model.loss(t(b[:, 0]), t(b[:, 1]), t(neg_m), t(neg_f))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n_batches += 1
        res = evaluate_model(model, data, "valid", k, all_train=all_train)
        score = res[valid_metric]
        history.append({"epoch": epoch, "loss": total / n_batches, **{f"valid_{a}": v for a, v in res.items()}})
        if score > best:
            best, best_epoch, wait = score, epoch, 0
            best_state = copy.deepcopy({a: v.detach().cpu() for a, v in model.state_dict().items()})
        else:
            wait += 1
        if verbose and (epoch % 10 == 0 or epoch == 1 or wait == 0):
            print(f"[{name} seed {seed}] epoch {epoch:4d}  loss {total / n_batches:.4f}  "
                  f"valid recall@50 {res['recall@50']:.4f}  crecall@50 {score:.4f}"
                  f"{'  *best*' if wait == 0 else ''}  ({time.time() - t0:.0f}s)")
        if wait >= patience:
            break
    model.load_state_dict({a: v.to(device) for a, v in best_state.items()})
    test = evaluate_model(model, data, "test", k, all_train=all_train)
    info = {"model": name, "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
            "best_valid_" + valid_metric: best, "train_seconds": time.time() - t0, **model_kw}
    info.pop("train_pairs", None)
    if verbose:
        print(f"[{name} seed {seed}] best epoch {best_epoch}; TEST recall@50 {test['recall@50']:.4f}  "
              f"ndcg@50 {test['ndcg@50']:.4f}  crecall@50 {test['crecall@50']:.4f}")
    return model, history, test, info
