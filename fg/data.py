"""Elliptic Bitcoin dataset: 203,769 transactions, 234,355 payment edges, 49 time steps.

Label: 1 = illicit, 0 = licit, -1 = unknown (~77 %). Features are anonymised: 93 local, 72 aggregated.
Edges only connect transactions inside the same time step.
"""
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "nodes.parquet"
HF = "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/"
FILES = ("elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv")
LOCAL = [f"l{i}" for i in range(93)]
AGG = [f"a{i}" for i in range(72)]
AF = LOCAL + AGG


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (RAW / f).exists():
            urllib.request.urlretrieve(HF + f, RAW / f)


def load():
    """Return (nodes, edges). nodes: txId, ts, y, AF columns; row position = graph node id."""
    download()
    edges = pd.read_csv(RAW / FILES[2])
    if CACHE.exists():
        return pd.read_parquet(CACHE), edges
    nodes = pd.read_csv(RAW / FILES[0], header=None, names=["txId", "ts"] + AF,
                        dtype={c: np.float32 for c in AF})
    cls = pd.read_csv(RAW / FILES[1]).set_index("txId")["class"].map({"1": 1, "2": 0, "unknown": -1})
    nodes.insert(2, "y", nodes["txId"].map(cls).astype(np.int8))
    nodes.to_parquet(CACHE, index=False)
    return nodes, edges
