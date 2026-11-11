"""Graph features from the payment graph. A[i, j] = 1 means j spends an output of i (i → j).

In-neighbours (inputs) are known when a transaction arrives; out-neighbours only appear later.
So the realtime model uses `in_*` only; the end-of-step batch model also uses `out_*`.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from fg.data import LOCAL

IN = [f"in_{c}" for c in LOCAL] + ["deg_in"]
OUT = [f"out_{c}" for c in LOCAL] + ["deg_out"]
S_IN = ["in_s_mean", "in_s_max"]
S_OUT = ["out_s_mean", "out_s_max"]


def adjacency(nodes: pd.DataFrame, edges: pd.DataFrame) -> sp.csr_matrix:
    pos = pd.Index(nodes["txId"])
    r, c = pos.get_indexer(edges["txId1"]), pos.get_indexer(edges["txId2"])
    keep = (r >= 0) & (c >= 0)
    n = len(nodes)
    return sp.csr_matrix((np.ones(keep.sum(), np.float32), (r[keep], c[keep])), shape=(n, n))


def _mean(A: sp.csr_matrix, M: np.ndarray):
    deg = np.asarray(A.sum(1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(A @ M) / deg[:, None], deg  # deg 0 → NaN, LightGBM treats as missing


def neighbor_features(nodes: pd.DataFrame, A: sp.csr_matrix) -> pd.DataFrame:
    L = nodes[LOCAL].to_numpy(np.float32)
    m_in, d_in = _mean(A.T.tocsr(), L)
    m_out, d_out = _mean(A, L)
    f = pd.DataFrame(np.hstack([m_in, m_out]).astype(np.float32), index=nodes.index,
                     columns=IN[:-1] + OUT[:-1])
    f["deg_in"], f["deg_out"] = d_in, d_out
    return f


def score_features(A: sp.csr_matrix, s: np.ndarray) -> pd.DataFrame:
    """Mean / max of first-stage fraud scores over in- and out-neighbours (collective classification)."""
    out = {}
    for name, M in (("in", A.T.tocsr()), ("out", A)):
        mean, deg = _mean(M, s[:, None])
        mx = np.asarray(M.multiply(s[None, :]).max(axis=1).todense()).ravel()
        out[f"{name}_s_mean"] = mean.ravel()
        out[f"{name}_s_max"] = np.where(deg > 0, mx, np.nan)
    return pd.DataFrame(out)


if __name__ == "__main__":
    nodes = pd.DataFrame({"txId": [10, 11, 12], **{c: [1.0, 3.0, 5.0] for c in LOCAL}})
    A = adjacency(nodes, pd.DataFrame({"txId1": [10, 11], "txId2": [12, 12]}))
    f = neighbor_features(nodes, A)
    assert f.loc[2, "in_l0"] == 2.0 and f.loc[2, "deg_in"] == 2 and np.isnan(f.loc[0, "in_l0"])
    s = score_features(A, np.array([0.2, 0.8, 0.1]))
    assert np.isclose(s.loc[2, "in_s_max"], 0.8) and np.isclose(s.loc[0, "out_s_mean"], 0.1)
    print("ok")
