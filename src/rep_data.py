# ---------------------------------------------------------------------------
# Data preparation for the DPGNN replication on Libimseti.
# Mirrors the protocol of the official CRRS / RecBole-PJF code:
#   * one row per (man, woman) pair; label = 1 means a mutual match
#   * direct = 0: the man acted first, direct = 1: the woman acted first
#   * random-order ("RO") split 80/10/10, grouped by man, RecBole's rounding rule
#   * train / valid / test targets keep only label = 1 rows (mutual matches)
#   * one-sided (label = 0) rows become extra "add" edges in DPGNN's graph,
#     taken from the whole file exactly as the official code does
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd


def load_libimseti(path):
    """Read the CRRS processed file and remap ids to 0..M-1 (men), 0..F-1 (women)."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    df = df.rename(columns={"male_id": "m_tok", "female_id": "f_tok"})
    m_tokens = np.sort(df["m_tok"].unique())
    f_tokens = np.sort(df["f_tok"].unique())
    df["m"] = np.searchsorted(m_tokens, df["m_tok"]).astype(np.int64)
    df["f"] = np.searchsorted(f_tokens, df["f_tok"]).astype(np.int64)
    df = df[["m", "f", "direct", "biexposure", "label"]].reset_index(drop=True)
    return df, len(m_tokens), len(f_tokens)


def _calcu_split_ids(tot, ratios):
    """RecBole's Dataset._calcu_split_ids: guarantees small groups still get valid/test rows."""
    cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
    cnt[0] = tot - sum(cnt[1:])
    for i in range(1, len(ratios)):
        if cnt[0] <= 1:
            break
        if 0 < ratios[-i] * tot < 1:
            cnt[-i] += 1
            cnt[0] -= 1
    return list(np.cumsum(cnt)[:-1])


def split_by_man(df, ratios=(0.8, 0.1, 0.1), seed=2020):
    """Shuffle all rows, then split each man's rows 80/10/10 (RecBole 'RS' + order 'RO' + group_by user)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    shuffled = df.iloc[order].reset_index(drop=True)
    part = np.empty(len(shuffled), dtype=np.int8)
    for _, idx in shuffled.groupby("m", sort=False).indices.items():
        cuts = _calcu_split_ids(len(idx), ratios)
        bounds = [0] + cuts + [len(idx)]
        for p in range(3):
            part[idx[bounds[p]:bounds[p + 1]]] = p
    shuffled["part"] = part
    return shuffled


class ReciprocalData:
    """Everything the models and the evaluator need, as plain numpy arrays."""

    def __init__(self, path, seed=2020):
        df, self.n_m, self.n_f = load_libimseti(path)
        self.raw = df
        sp = split_by_man(df, seed=seed)
        self.split = sp
        pos = sp[sp.label == 1]
        self.train_pos = pos[pos.part == 0][["m", "f"]].to_numpy()
        self.valid_pos = pos[pos.part == 1][["m", "f"]].to_numpy()
        self.test_pos = pos[pos.part == 2][["m", "f"]].to_numpy()
        # one-sided interactions (no match). The official code builds these from the full file.
        fail = df[df.label == 0]
        self.add_m = fail[fail.direct == 0][["m", "f"]].to_numpy()  # man liked woman, no match
        self.add_f = fail[fail.direct == 1][["m", "f"]].to_numpy()  # woman liked man, no match
        # training rows of all labels (used by the LightGCN baseline, CRRS sets REMAIN_NEG=True for it)
        self.train_all = sp[sp.part == 0][["m", "f"]].to_numpy()
        # sets for negative sampling: a sampled negative must not be a training match
        self.m2f_train = [set() for _ in range(self.n_m)]
        self.f2m_train = [set() for _ in range(self.n_f)]
        for m, f in self.train_pos:
            self.m2f_train[m].add(f)
            self.f2m_train[f].add(m)

    def history(self, phase, all_train=False):
        """Pairs hidden from ranking: train matches for validation; train + valid matches for test.
        all_train=True also hides one-sided training interactions, as the official code does for models
        trained on them (LightGCN with REMAIN_NEG: True, whose RecBole history holds every training row)."""
        base = self.train_all if all_train else self.train_pos
        return base if phase == "valid" else np.vstack([base, self.valid_pos])

    def targets(self, phase):
        return self.valid_pos if phase == "valid" else self.test_pos

    def summary(self):
        return {
            "men": self.n_m, "women": self.n_f, "interactions": len(self.raw),
            "mutual_matches": int(self.raw.label.sum()),
            "train_matches": len(self.train_pos), "valid_matches": len(self.valid_pos),
            "test_matches": len(self.test_pos),
            "one_sided_by_men": len(self.add_m), "one_sided_by_women": len(self.add_f),
        }


def sample_negatives(data, m_batch, f_batch, rng):
    """Uniform negatives: a woman the man has not matched in training, and a man the woman has not matched."""
    neg_f = rng.integers(0, data.n_f, size=len(m_batch))
    neg_m = rng.integers(0, data.n_m, size=len(f_batch))
    for i in range(len(m_batch)):
        s = data.m2f_train[m_batch[i]]
        while neg_f[i] in s:
            neg_f[i] = rng.integers(0, data.n_f)
    for i in range(len(f_batch)):
        s = data.f2m_train[f_batch[i]]
        while neg_m[i] in s:
            neg_m[i] = rng.integers(0, data.n_m)
    return neg_m, neg_f
