# ---------------------------------------------------------------------------
# Models, written in plain PyTorch (torch.sparse instead of torch_geometric) so they
# run on a fresh Colab runtime with no extra installs.
#   DPGNN    : faithful re-implementation of recbole_pjf/model/dpgnn.py (official code)
#   LightGCN : RecBole LightGCN as configured in CRRS (trains on all interactions)
#   BPRMF    : RecBole BPR as configured in CRRS
# Node layout of DPGNN's graph (same as the official code):
#   [ men-active | women-passive | men-passive | women-active ]
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bpr_loss(pos, neg, gamma=1e-10):
    """RecBole BPRLoss."""
    return -torch.log(gamma + torch.sigmoid(pos - neg)).mean()


def emb_loss(*embeddings):
    """RecBole EmbLoss with require_pow=False: sum of L2 norms divided by the batch size."""
    loss = torch.zeros(1, device=embeddings[-1].device)
    for e in embeddings:
        loss = loss + torch.norm(e, p=2)
    return loss / embeddings[-1].shape[0]


def sym_norm_adj(rows, cols, n_nodes, device):
    """D^-1/2 A D^-1/2 from an edge list in which every edge already appears in both directions.
    Degree = number of listed edges per node, exactly as torch_geometric.utils.degree in the official code."""
    rows = torch.as_tensor(rows, dtype=torch.long)
    cols = torch.as_tensor(cols, dtype=torch.long)
    deg = torch.bincount(rows, minlength=n_nodes).float()
    norm = 1.0 / torch.sqrt(torch.where(deg == 0, torch.ones(1), deg))
    w = norm[rows] * norm[cols]
    # A[target, source]: messages flow source -> target and are summed (aggr='add')
    adj = torch.sparse_coo_tensor(torch.stack([cols, rows]), w, (n_nodes, n_nodes)).coalesce()
    return adj.to(device)


def both_ways(a, b):
    return np.concatenate([a, b]), np.concatenate([b, a])


class DPGNN(nn.Module):
    """Dual-Perspective GNN (Yang et al., RecSys 2022) for men (side A) and women (side B)."""

    def __init__(self, data, device, dim=128, n_layers=3, reg_weight=1e-4, mutual_weight=0.05,
                 temperature=0.2, use_add_edges=True, use_mutual_loss=True):
        super().__init__()
        self.n_u, self.n_i, self.device = data.n_m, data.n_f, device
        self.n_layers, self.reg_weight = n_layers, reg_weight
        self.mutual_weight = mutual_weight if use_mutual_loss else 0.0
        self.temperature = temperature
        self.user_a = nn.Embedding(self.n_u, dim)   # man as chooser
        self.item_a = nn.Embedding(self.n_i, dim)   # woman as chooser
        self.user_p = nn.Embedding(self.n_u, dim)   # man as the one chosen
        self.item_p = nn.Embedding(self.n_i, dim)   # woman as the one chosen
        for emb in (self.user_a, self.item_a, self.user_p, self.item_p):
            nn.init.xavier_normal_(emb.weight)
        self.adj = self._build_graph(data, use_add_edges)

    def _build_graph(self, data, use_add_edges):
        n_u, n_i = self.n_u, self.n_i
        n_all = n_u + n_i
        R, C = [], []

        def add(r, c):
            rr, cc = both_ways(r, c)
            R.append(rr); C.append(cc)

        # mutual matches (training only): man-active <-> woman-passive and man-passive <-> woman-active
        m, f = data.train_pos[:, 0], data.train_pos[:, 1] + n_u
        add(m, f)
        add(m + n_all, f + n_all)
        if use_add_edges:
            # one-sided interest by the man: man-active <-> woman-passive
            m, f = data.add_m[:, 0], data.add_m[:, 1] + n_u
            add(m, f)
            # one-sided interest by the woman: woman-active <-> man-passive
            m, f = data.add_f[:, 0], data.add_f[:, 1] + n_u
            add(m + n_all, f + n_all)
        # self edges linking each person's two nodes
        u = np.arange(n_u); i = np.arange(n_i) + n_u
        add(u, u + n_all)
        add(i, i + n_all)
        return sym_norm_adj(np.concatenate(R), np.concatenate(C), 2 * n_all, self.device)

    def ego(self):
        return torch.cat([self.user_a.weight, self.item_p.weight, self.user_p.weight, self.item_a.weight], 0)

    def propagate(self):
        x = self.ego()
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x)
            layers.append(x)
        out = torch.stack(layers, 1).mean(1)
        return torch.split(out, [self.n_u, self.n_i, self.n_u, self.n_i])  # u_a, i_p, u_p, i_a

    def _info_nce(self, a, p):
        a, p = F.normalize(a, dim=1), F.normalize(p, dim=1)
        sim = a @ p.T
        mask = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        logits = torch.cat([sim[mask].view(sim.shape[0], -1), sim[~mask].view(sim.shape[0], -1)], 1)
        labels = torch.zeros(sim.shape[0], dtype=torch.long, device=sim.device)
        return F.cross_entropy(logits / self.temperature, labels)

    def loss(self, m, f, neg_m, neg_f):
        u_a, i_p, u_p, i_a = self.propagate()
        r_pos = (u_a[m] * i_p[f]).sum(1)            # man's interest in woman
        s_pos = (u_p[m] * i_a[f]).sum(1)            # woman's interest in man
        r_neg1 = (u_a[m] * i_p[neg_f]).sum(1)
        s_neg1 = (u_p[m] * i_a[neg_f]).sum(1)
        r_neg2 = (u_a[neg_m] * i_p[f]).sum(1)
        s_neg2 = (u_p[neg_m] * i_a[f]).sum(1)
        loss = bpr_loss(2 * r_pos + 2 * s_pos, r_neg1 + s_neg1 + r_neg2 + s_neg2)
        loss = loss + self.reg_weight * emb_loss(
            self.user_a(m), self.user_p(m), self.item_a(f), self.item_p(f),
            self.item_a(neg_f), self.item_p(neg_f), self.user_a(neg_m), self.user_p(neg_m))
        if self.mutual_weight > 0:
            loss = loss + self.mutual_weight * self._info_nce(self.user_a.weight[m], self.user_p.weight[m])
            loss = loss + self.mutual_weight * self._info_nce(self.item_a.weight[f], self.item_p.weight[f])
        return loss

    @torch.no_grad()
    def full_scores(self, mode="both"):
        """[n_men, n_women] scores. mode: 'both' (official), 'man_side' or 'woman_side' (analysis only)."""
        u_a, i_p, u_p, i_a = self.propagate()
        if mode == "man_side":
            return u_a @ i_p.T
        if mode == "woman_side":
            return u_p @ i_a.T
        return u_a @ i_p.T + u_p @ i_a.T


class LightGCN(nn.Module):
    """RecBole LightGCN (He et al., SIGIR 2020) as run in CRRS: one node per person."""

    def __init__(self, data, device, dim=128, n_layers=3, reg_weight=5e-4, train_pairs=None):
        super().__init__()
        self.n_u, self.n_i, self.device = data.n_m, data.n_f, device
        self.n_layers, self.reg_weight = n_layers, reg_weight
        self.user = nn.Embedding(self.n_u, dim)
        self.item = nn.Embedding(self.n_i, dim)
        nn.init.xavier_uniform_(self.user.weight); nn.init.xavier_uniform_(self.item.weight)
        pairs = np.unique(train_pairs, axis=0)
        r, c = both_ways(pairs[:, 0], pairs[:, 1] + self.n_u)
        self.adj = sym_norm_adj(r, c, self.n_u + self.n_i, device)

    def propagate(self):
        x = torch.cat([self.user.weight, self.item.weight], 0)
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x)
            layers.append(x)
        out = torch.stack(layers, 1).mean(1)
        return torch.split(out, [self.n_u, self.n_i])

    def loss(self, m, f, neg_m, neg_f):
        u, i = self.propagate()
        loss = bpr_loss((u[m] * i[f]).sum(1), (u[m] * i[neg_f]).sum(1))
        return loss + self.reg_weight * emb_loss(self.user(m), self.item(f), self.item(neg_f))

    @torch.no_grad()
    def full_scores(self, mode="both"):
        u, i = self.propagate()
        return u @ i.T


class BPRMF(nn.Module):
    """RecBole BPR (Rendle et al., UAI 2009) as run in CRRS."""

    def __init__(self, data, device, dim=128, **_):
        super().__init__()
        self.user = nn.Embedding(data.n_m, dim)
        self.item = nn.Embedding(data.n_f, dim)
        nn.init.xavier_normal_(self.user.weight); nn.init.xavier_normal_(self.item.weight)

    def loss(self, m, f, neg_m, neg_f):
        u = self.user(m)
        return bpr_loss((u * self.item(f)).sum(1), (u * self.item(neg_f)).sum(1))

    @torch.no_grad()
    def full_scores(self, mode="both"):
        return self.user.weight @ self.item.weight.T
