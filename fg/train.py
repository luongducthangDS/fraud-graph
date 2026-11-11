"""Temporal backtest on Elliptic (train steps 1–34, test 35–49) and serving artifacts.

python -m fg.train
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from fg import data, graph

ART = data.ROOT / "artifacts"
TRAIN_END, SHUTDOWN = 34, 43  # step 43: dark-market shutdown, known regime change in Elliptic
REALTIME = data.AF + graph.IN
BATCH = REALTIME + graph.OUT
N_TREES = 300  # tests lower this


def gbm(seed=0, n_jobs=1):  # ponytail: 1 thread avoids OpenMP oversubscription on a shared box; raise on a dedicated one
    return LGBMClassifier(n_estimators=N_TREES, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                          colsample_bytree=0.5, subsample=0.8, subsample_freq=1, random_state=seed,
                          n_jobs=n_jobs, verbose=-1)


def metrics(y, s, thr=0.5):
    p, r, f, _ = precision_recall_fscore_support(y, s >= thr, average="binary", zero_division=0)
    return {"pr_auc": round(average_precision_score(y, s), 4), "precision": round(p, 4),
            "recall": round(r, 4), "f1": round(f, 4)}


def best_threshold(y, s):
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[np.argmax([metrics(y, s, t)["f1"] for t in grid])])


def oof_fit(F, cols, y, ts, seed=0, folds=5):
    """Out-of-fold scores on train steps (time-block folds), final-model scores on test steps.

    Edges stay inside one step, so a fold never sees its own neighbours' labels.
    """
    tr, lab = ts <= TRAIN_END, y >= 0
    s = np.full(len(F), np.nan)
    for block in np.array_split(np.arange(1, TRAIN_END + 1), folds):
        hold = np.isin(ts, block)
        m = gbm(seed).fit(F.loc[tr & lab & ~hold, cols], y[tr & lab & ~hold])
        s[hold] = m.predict_proba(F.loc[hold, cols])[:, 1]
    final = gbm(seed).fit(F.loc[tr & lab, cols], y[tr & lab])
    s[~tr] = final.predict_proba(F.loc[~tr, cols])[:, 1]
    return s, final


def two_stage(F, A, cols, s_cols, y, ts, seed=0):
    s1, m1 = oof_fit(F, cols, y, ts, seed)
    F2 = pd.concat([F[cols], graph.score_features(A, s1)[s_cols].set_axis(F.index)], axis=1)
    s2, m2 = oof_fit(F2, cols + s_cols, y, ts, seed)
    return s1, s2, m1, m2


def self_training(F, cols, y, ts, frac, seed, rounds=3, hi=0.9, lo=0.02):
    """Low-label regime: keep `frac` of train labels, pseudo-label confident nodes from the unlabelled pool."""
    rng = np.random.default_rng(seed)
    tr = ts <= TRAIN_END
    L = tr & (y >= 0) & (rng.random(len(y)) < frac)
    pool = tr & ~L  # unknown nodes + train nodes whose label we hide
    test = (ts > TRAIN_END) & (y >= 0)
    m = gbm(seed).fit(F.loc[L, cols], y[L])
    out = {"labels": int(L.sum()), "supervised": metrics(y[test], m.predict_proba(F.loc[test, cols])[:, 1])}
    ratio = (y[L] == 0).sum() / max((y[L] == 1).sum(), 1)
    Xp = F.loc[pool, cols]
    for _ in range(rounds):
        p = m.predict_proba(Xp)[:, 1]
        pos = np.flatnonzero(p >= hi)
        neg = np.flatnonzero(p <= lo)
        neg = rng.choice(neg, min(len(neg), int(ratio * max(len(pos), 1))), replace=False)  # keep class ratio
        X = pd.concat([F.loc[L, cols], Xp.iloc[pos], Xp.iloc[neg]])
        yy = np.concatenate([y[L], np.ones(len(pos), int), np.zeros(len(neg), int)])
        m = gbm(seed).fit(X, yy)
    out["self_training"] = metrics(y[test], m.predict_proba(F.loc[test, cols])[:, 1])
    out["pseudo_labels_last_round"] = {"illicit": int(len(pos)), "licit": int(len(neg))}
    return out


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    b = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((b - a) * np.log(b / a)))


def monitoring(y, s, ts, thr):
    ref = s[(ts <= TRAIN_END)]
    rows = []
    for t in range(TRAIN_END + 1, int(ts.max()) + 1):
        m = ts == t
        lab = m & (y >= 0)
        rows.append({"step": t, "n": int(m.sum()), "alert_rate": round(float((s[m] >= thr).mean()), 4),
                     "psi": round(psi(ref, s[m]), 4), **metrics(y[lab], s[lab], thr)})
    return rows


def main(nodes=None, edges=None, out: Path = ART, seed=0, semi_fracs=(0.02, 0.1), semi_seeds=(0, 1, 2)):
    t0 = time.time()
    if nodes is None:
        nodes, edges = data.load()
    out.mkdir(parents=True, exist_ok=True)
    y, ts = nodes["y"].to_numpy(int), nodes["ts"].to_numpy(int)
    A = graph.adjacency(nodes, edges)
    F = pd.concat([nodes[data.AF], graph.neighbor_features(nodes, A)], axis=1)
    test = (ts > TRAIN_END) & (y >= 0)
    tr_lab = (ts <= TRAIN_END) & (y >= 0)
    post = test & (ts >= SHUTDOWN)
    rep = {"nodes": len(nodes), "edges": int(A.nnz), "train_labels": int(tr_lab.sum()),
           "test_labels": int(test.sum()), "test_illicit_rate": round(float(y[test].mean()), 4)}

    # unsupervised anomaly baseline: never sees a label
    iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=2).fit(F.loc[ts <= TRAIN_END, data.AF])
    s_iso = -iso.score_samples(F.loc[test, data.AF])
    k = int(y[test].sum())
    res = {"isolation_forest_AF": {"pr_auc": round(average_precision_score(y[test], s_iso), 4),
                                   "precision_at_k": round(float(y[test][np.argsort(-s_iso)[:k]].mean()), 4)}}
    rf = RandomForestClassifier(n_estimators=200, max_features=50, random_state=seed, n_jobs=2)
    rf.fit(F.loc[tr_lab, data.AF], y[tr_lab])
    res["random_forest_AF (paper baseline)"] = metrics(y[test], rf.predict_proba(F.loc[test, data.AF])[:, 1])

    for name, cols in (("lgbm_LF", data.LOCAL), ("lgbm_AF", data.AF)):
        s, _ = oof_fit(F, cols, y, ts, seed)
        res[name] = metrics(y[test], s[test])
    thr = {}
    for name, cols, s_cols in (("realtime", REALTIME, graph.S_IN), ("batch", BATCH, graph.S_IN + graph.S_OUT)):
        s1, s2, m1, m2 = two_stage(F, A, cols, s_cols, y, ts, seed)
        thr[name] = best_threshold(y[tr_lab], s2[tr_lab])  # from out-of-fold scores, never from test
        res[f"lgbm_{name}_stage1 (AF+NE)"] = metrics(y[test], s1[test])
        res[f"lgbm_{name}_stage2 (AF+NE+S)"] = metrics(y[test], s2[test])
        res[f"lgbm_{name}_stage2 @oof-threshold"] = {**metrics(y[test], s2[test], thr[name]), "threshold": thr[name]}
        res[f"lgbm_{name}_stage2 f1 steps<{SHUTDOWN} / >={SHUTDOWN}"] = [
            metrics(y[test & ~post], s2[test & ~post])["f1"], metrics(y[post], s2[post])["f1"]]
        joblib.dump({"stage1": m1, "stage2": m2, "cols": cols, "s_cols": s_cols, "threshold": thr[name]},
                    out / f"{name}.joblib")
        if name == "realtime":
            rep["monitoring"] = monitoring(y, s2, ts, thr[name])
            # online store for the realtime API: local features + stage-1 score per known transaction
            store = nodes[["txId", "ts"] + data.LOCAL].assign(s1=s1.astype(np.float32))
            store.to_parquet(out / "store.parquet", index=False)
        pd.DataFrame({"txId": nodes["txId"], "ts": ts, "y": y, "score": s2}).to_parquet(out / f"scores_{name}.parquet", index=False)
    rep["models"] = res

    rep["semi_supervised_AF"] = {
        str(f): {"per_seed": [self_training(F, data.AF, y, ts, f, s) for s in semi_seeds]} for f in semi_fracs}
    for f, v in rep["semi_supervised_AF"].items():
        for kind in ("supervised", "self_training"):
            v[f"mean_{kind}"] = {m: round(float(np.mean([r[kind][m] for r in v["per_seed"]])), 4)
                                 for m in ("pr_auc", "f1")}
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != "monitoring"}, indent=2))
