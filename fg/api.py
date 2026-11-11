"""Realtime scoring API.

A new transaction arrives with its 165 features and the ids of the transactions it spends from (inputs).
Graph features come from the online store (local features + stage-1 score of every known transaction);
the scored transaction is then written back so later transactions can use it as a neighbour.
"""
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fg import data, graph

ART = Path(os.environ.get("FG_ARTIFACTS", Path(__file__).resolve().parents[1] / "artifacts"))
THRESHOLD = os.environ.get("FG_THRESHOLD")  # override the out-of-fold F1 threshold, e.g. to fit an alert budget
app = FastAPI(title="Fraud graph scoring")
S: dict = {}


def state():
    if not S:
        store = pd.read_parquet(ART / "store.parquet")
        S["pos"] = {int(t): i for i, t in enumerate(store["txId"])}
        S["local"] = [store[data.LOCAL].to_numpy(np.float32)]  # list of blocks; new rows appended as blocks
        S["s1"] = list(store["s1"].to_numpy(np.float32))
        S["model"] = joblib.load(ART / "realtime.joblib")
        S["report"] = json.loads((ART / "report.json").read_text())
    return S


def local_row(i):
    # ponytail: blocks of new rows are 1-row arrays; compact into one array if the store grows large
    blocks, base = S["local"], len(S["local"][0])
    return blocks[0][i] if i < base else blocks[i - base + 1][0]


class Tx(BaseModel):
    tx_id: int
    features: list[float]  # 165 values in Elliptic order: 93 local then 72 aggregated
    inputs: list[int] = []


def features(tx: Tx):
    s, m = state(), S["model"]
    if len(tx.features) != len(data.AF):
        raise HTTPException(422, f"expected {len(data.AF)} features, got {len(tx.features)}")
    found = [S["pos"][i] for i in tx.inputs if i in S["pos"]]
    nb = np.array([local_row(i) for i in found]) if found else np.full((1, len(data.LOCAL)), np.nan)
    row = pd.DataFrame([list(tx.features) + list(nb.mean(0)) + [len(tx.inputs)]], columns=data.AF + graph.IN)
    s1 = float(m["stage1"].predict_proba(row[m["cols"]])[0, 1])
    ns = np.array([S["s1"][i] for i in found], float)
    row["in_s_mean"], row["in_s_max"] = (ns.mean(), ns.max()) if found else (np.nan, np.nan)
    return row, s1, len(found)


@app.get("/health")
def health():
    return {"known_transactions": len(state()["pos"]), "threshold": threshold()}


def threshold():
    return float(THRESHOLD) if THRESHOLD else state()["model"]["threshold"]


@app.post("/score")
def score(tx: Tx):
    t0 = time.perf_counter()
    row, s1, found = features(tx)
    m = S["model"]
    p = float(m["stage2"].predict_proba(row[m["cols"] + m["s_cols"]])[0, 1])
    if tx.tx_id not in S["pos"]:  # write back so later transactions see this one as an input
        S["pos"][tx.tx_id] = len(S["s1"])
        S["local"].append(np.asarray(tx.features[:len(data.LOCAL)], np.float32)[None, :])
        S["s1"].append(s1)
    return {"tx_id": tx.tx_id, "score": round(p, 5), "alert": p >= threshold(), "threshold": threshold(),
            "inputs_found": found, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.get("/monitoring")
def monitoring():
    """Per test time step: alert rate, score PSI vs train, precision / recall where labels exist."""
    return state()["report"]["monitoring"]
