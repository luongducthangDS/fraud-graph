"""End-of-step batch job: rescore whole time steps with in + out edges (the batch model).

python -m fg.batch 43 44 45      # writes artifacts/batch_scores.parquet
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fg import data, graph
from fg.train import ART


def score_steps(nodes, edges, steps, art: Path = ART) -> pd.DataFrame:
    b = joblib.load(art / "batch.joblib")
    A = graph.adjacency(nodes, edges)
    F = pd.concat([nodes[data.AF], graph.neighbor_features(nodes, A)], axis=1)
    m = nodes["ts"].isin(steps).to_numpy()
    s1 = np.zeros(len(nodes))  # edges stay inside a step, so only rows in `steps` matter
    s1[m] = b["stage1"].predict_proba(F.loc[m, b["cols"]])[:, 1]
    F2 = pd.concat([F[b["cols"]], graph.score_features(A, s1)[b["s_cols"]].set_axis(F.index)], axis=1)
    p = b["stage2"].predict_proba(F2.loc[m, b["cols"] + b["s_cols"]])[:, 1]
    return pd.DataFrame({"txId": nodes.loc[m, "txId"].to_numpy(), "ts": nodes.loc[m, "ts"].to_numpy(),
                         "score": p, "alert": p >= b["threshold"]})


if __name__ == "__main__":
    nodes, edges = data.load()
    out = score_steps(nodes, edges, [int(s) for s in sys.argv[1:]])
    out.to_parquet(ART / "batch_scores.parquet", index=False)
    print(out.groupby("ts").agg(n=("score", "size"), alerts=("alert", "sum")))
