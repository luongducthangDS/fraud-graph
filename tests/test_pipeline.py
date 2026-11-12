"""Synthetic Elliptic-like graph: leakage, train/serve parity (realtime + batch), API write-back. Runs in CI."""
import importlib

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fg import batch, data, graph, train


def synth(n=80, steps=49, seed=0):
    """Illicit nodes shift a few local features and mostly spend from other illicit nodes (homophily)."""
    rng = np.random.default_rng(seed)
    nodes, edges, next_id = [], [], 1000
    for t in range(1, steps + 1):
        ids = np.arange(next_id, next_id + n)
        next_id += n
        bad = rng.random(n) < 0.12
        X = rng.normal(size=(n, len(data.AF))).astype(np.float32)
        X[bad, :5] += 1.2
        y = np.where(rng.random(n) < 0.6, -1, bad.astype(int))
        for j in range(5, n):
            same = np.flatnonzero(bad[:j] == bad[j])
            pool = same if len(same) and rng.random() < 0.8 else np.arange(j)
            for i in rng.choice(pool, min(len(pool), rng.integers(1, 3)), replace=False):
                edges.append((ids[i], ids[j]))
        nodes.append(pd.DataFrame(X, columns=data.AF).assign(txId=ids, ts=t, y=y))
    nodes = pd.concat(nodes, ignore_index=True)
    return nodes[["txId", "ts", "y"] + data.AF], pd.DataFrame(edges, columns=["txId1", "txId2"])


@pytest.fixture(scope="module", autouse=True)
def small_models():
    train.N_TREES = 50


@pytest.fixture(scope="module")
def g():
    return synth()


@pytest.fixture(scope="module")
def trained(g, tmp_path_factory):
    out = tmp_path_factory.mktemp("art")
    return train.main(*g, out=out, semi_fracs=(0.3,), semi_seeds=(0,)), out


def test_test_labels_never_used(g):
    nodes, _ = g
    y, ts = nodes["y"].to_numpy(int), nodes["ts"].to_numpy(int)
    s, _ = train.oof_fit(nodes, data.AF, y, ts)
    y2 = y.copy()
    test = (ts > train.TRAIN_END) & (y >= 0)
    y2[test] = 1 - y2[test]
    s2, _ = train.oof_fit(nodes, data.AF, y2, ts)
    np.testing.assert_allclose(s, s2)


def test_report_and_signal(trained):
    rep, _ = trained
    assert rep["models"]["lgbm_realtime_stage2 (AF+NE+S)"]["pr_auc"] > 0.4  # base rate ≈ 0.12
    assert len(rep["monitoring"]) == 49 - train.TRAIN_END
    assert "self_training" in rep["semi_supervised_AF"]["0.3"]["per_seed"][0]


def client(out, monkeypatch):
    monkeypatch.setenv("FG_ARTIFACTS", str(out))
    from fg import api
    return TestClient(importlib.reload(api).app)


def test_realtime_api_matches_offline(g, trained, monkeypatch):
    nodes, edges = g
    _, out = trained
    c = client(out, monkeypatch)
    offline = pd.read_parquet(out / "scores_realtime.parquet").set_index("txId")["score"]
    for tx in nodes.loc[nodes["ts"] == 45, "txId"].iloc[[3, 20, 60]]:
        row = nodes.loc[nodes["txId"] == tx, data.AF].iloc[0].tolist()
        inputs = edges.loc[edges["txId2"] == tx, "txId1"].tolist()
        r = c.post("/score", json={"tx_id": int(tx), "features": row, "inputs": inputs}).json()
        assert r["inputs_found"] == len(inputs)
        assert r["score"] == pytest.approx(offline[tx], abs=1e-3)


def test_realtime_write_back_and_validation(g, trained, monkeypatch):
    _, out = trained
    c = client(out, monkeypatch)
    feats = [0.0] * len(data.AF)
    assert c.post("/score", json={"tx_id": 1, "features": feats}).json()["inputs_found"] == 0
    assert c.post("/score", json={"tx_id": 2, "features": feats, "inputs": [1]}).json()["inputs_found"] == 1
    assert c.post("/score", json={"tx_id": 3, "features": [0.0]}).status_code == 422
    assert len(c.get("/monitoring").json()) == 49 - train.TRAIN_END


def test_batch_job_matches_offline(g, trained):
    nodes, edges = g
    _, out = trained
    got = batch.score_steps(nodes, edges, [40, 41], art=out).set_index("txId")["score"]
    want = pd.read_parquet(out / "scores_batch.parquet").set_index("txId")["score"].loc[got.index]
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_api_safety_and_observability(g, trained, monkeypatch):
    _, out = trained
    c = client(out, monkeypatch)
    feats = [0.0] * len(data.AF)
    r = c.post("/score", json={"tx_id": 10, "features": feats}, headers={"x-request-id": "abc"})
    assert r.headers["x-request-id"] == "abc"
    body = r.json()
    assert body["action"] == ("analyst_review" if body["alert"] else "pass") and body["model_version"]
    assert c.post("/score", json={"tx_id": -1, "features": feats}).status_code == 422
    assert c.post("/score", json={"tx_id": 11, "features": feats, "inputs": list(range(2000))}).status_code == 422
    m = c.get("/metrics").json()
    assert m["scored_recent"] >= 1 and 0 <= m["alert_rate_recent"] <= 1 and m["latency_ms"]["POST /score"]["n"] >= 1

    from fg import api
    monkeypatch.setattr(api, "API_KEY", "k3y")
    assert c.get("/health").status_code == 200
    assert c.post("/score", json={"tx_id": 12, "features": feats}).status_code == 401
    assert c.post("/score", json={"tx_id": 12, "features": feats}, headers={"X-API-Key": "k3y"}).status_code == 200


def test_missing_artifacts_is_503(tmp_path, monkeypatch):
    assert client(tmp_path, monkeypatch).get("/health").status_code == 503


def test_gate():
    import json
    from fg import gate
    base = json.loads(gate.BASELINE.read_text(encoding="utf-8"))
    rep = json.loads((gate.ROOT / "artifacts" / "report.json").read_text())
    assert gate.check(rep, base) == []  # report đã commit phải qua gate
    worse = json.loads(json.dumps(rep))
    rt = worse["models"]["lgbm_realtime_stage2 (AF+NE+S)"]
    rt["pr_auc"] = worse["models"]["random_forest_AF (paper baseline)"]["pr_auc"] - 0.01
    fails = gate.check(worse, base)
    assert any("baseline" in f for f in fails) and any("không thắng" in f for f in fails)
