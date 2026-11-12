"""Realtime scoring API.

A new transaction arrives with its 165 features and the ids of the transactions it spends from (inputs).
Graph features come from the online store (local features + stage-1 score of every known transaction);
the scored transaction is then written back so later transactions can use it as a neighbour.

API không tự chặn giao dịch: điểm >= ngưỡng trả `action: analyst_review` để chuyên viên AML quyết định.
Env: FG_API_KEY (set -> mọi endpoint trừ /health cần header X-API-Key), FG_THRESHOLD (ghi đè ngưỡng).
Mỗi request ghi một dòng log JSON (request_id, route, status, latency, model_version; /score thêm score, alert).
"""
import hashlib
import hmac
import json
import logging
import math
import os
import statistics
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from fg import data, graph

ART = Path(os.environ.get("FG_ARTIFACTS", Path(__file__).resolve().parents[1] / "artifacts"))
THRESHOLD = os.environ.get("FG_THRESHOLD")  # override the out-of-fold F1 threshold, e.g. to fit an alert budget
API_KEY = os.environ.get("FG_API_KEY")
MAX_INPUTS = 1000  # Elliptic: in-degree lớn nhất vài trăm
app = FastAPI(title="Fraud graph scoring")
S: dict = {}
log = logging.getLogger("fg.api")
logging.basicConfig(level=logging.INFO, format="%(message)s")
# ponytail: metrics trong bộ nhớ một tiến trình; nhiều worker thì đẩy sang Prometheus
LAT: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
COUNT: dict[str, int] = defaultdict(int)
ALERTS: deque = deque(maxlen=1000)  # tỉ lệ cảnh báo gần đây: tín hiệu drift thô (sập sau step 43 làm nó rơi)


class ArtifactsMissing(Exception):
    pass


def state():
    if not S:
        try:
            store = pd.read_parquet(ART / "store.parquet")
            S["pos"] = {int(t): i for i, t in enumerate(store["txId"])}
            S["local"] = [store[data.LOCAL].to_numpy(np.float32)]  # list of blocks; new rows appended as blocks
            S["s1"] = list(store["s1"].to_numpy(np.float32))
            S["model"] = joblib.load(ART / "realtime.joblib")
            S["report"] = json.loads((ART / "report.json").read_text())
            S["version"] = hashlib.sha256((ART / "realtime.joblib").read_bytes()).hexdigest()[:12]
        except (FileNotFoundError, OSError) as e:
            S.clear()
            raise ArtifactsMissing(str(e)) from e
    return S


@app.exception_handler(ArtifactsMissing)
def _missing(_: Request, e: ArtifactsMissing):
    return JSONResponse({"detail": f"model artifacts not loaded, run `python -m fg.train`: {e}"}, status_code=503)


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.rid, request.state.extra = rid, {}
    t0 = time.perf_counter()
    resp = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    route = request.scope.get("route")
    key = f"{request.method} {route.path if route else request.url.path}"
    COUNT[f"{key} {resp.status_code}"] += 1
    LAT[key].append(ms)
    resp.headers["x-request-id"] = rid
    log.info(json.dumps({"request_id": rid, "route": key, "status": resp.status_code, "latency_ms": ms,
                         "model_version": S.get("version"), **request.state.extra}))
    return resp


def auth(x_api_key: str | None = Header(None)):
    if API_KEY and not (x_api_key and hmac.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(401, "invalid or missing X-API-Key")


def local_row(i):
    # ponytail: blocks of new rows are 1-row arrays; compact into one array if the store grows large
    blocks, base = S["local"], len(S["local"][0])
    return blocks[0][i] if i < base else blocks[i - base + 1][0]


class Tx(BaseModel):
    tx_id: int = Field(ge=0)
    features: list[float]  # 165 values in Elliptic order: 93 local then 72 aggregated
    inputs: list[int] = Field(default=[], max_length=MAX_INPUTS)


def features(tx: Tx):
    s, m = state(), S["model"]
    if len(tx.features) != len(data.AF):
        raise HTTPException(422, f"expected {len(data.AF)} features, got {len(tx.features)}")
    if not all(math.isfinite(v) for v in tx.features):
        raise HTTPException(422, "features must be finite")
    found = [S["pos"][i] for i in tx.inputs if i in S["pos"]]
    nb = np.array([local_row(i) for i in found]) if found else np.full((1, len(data.LOCAL)), np.nan)
    row = pd.DataFrame([list(tx.features) + list(nb.mean(0)) + [len(tx.inputs)]], columns=data.AF + graph.IN)
    s1 = float(m["stage1"].predict_proba(row[m["cols"]])[0, 1])
    ns = np.array([S["s1"][i] for i in found], float)
    row["in_s_mean"], row["in_s_max"] = (ns.mean(), ns.max()) if found else (np.nan, np.nan)
    return row, s1, len(found)


@app.get("/health")
def health():
    return {"known_transactions": len(state()["pos"]), "threshold": threshold(), "model_version": S["version"]}


def threshold():
    return float(THRESHOLD) if THRESHOLD else state()["model"]["threshold"]


@app.post("/score", dependencies=[Depends(auth)])
def score(tx: Tx, request: Request):
    t0 = time.perf_counter()
    row, s1, found = features(tx)
    m = S["model"]
    p = float(m["stage2"].predict_proba(row[m["cols"] + m["s_cols"]])[0, 1])
    if tx.tx_id not in S["pos"]:  # write back so later transactions see this one as an input
        S["pos"][tx.tx_id] = len(S["s1"])
        S["local"].append(np.asarray(tx.features[:len(data.LOCAL)], np.float32)[None, :])
        S["s1"].append(s1)
    alert = p >= threshold()
    ALERTS.append(alert)
    request.state.extra = {"tx_id": tx.tx_id, "score": round(p, 5), "alert": alert, "inputs_found": found}
    return {"tx_id": tx.tx_id, "score": round(p, 5), "alert": alert, "threshold": threshold(),
            "action": "analyst_review" if alert else "pass", "inputs_found": found,
            "model_version": S["version"], "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.get("/metrics", dependencies=[Depends(auth)])
def metrics():
    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, math.ceil(q * len(xs)) - 1)]
    return {"model_version": S.get("version"), "requests": dict(COUNT),
            "alert_rate_recent": round(sum(ALERTS) / len(ALERTS), 4) if ALERTS else None,
            "scored_recent": len(ALERTS),
            "latency_ms": {k: {"n": len(v), "p50": statistics.median(v), "p95": pct(v, 0.95)}
                           for k, v in LAT.items() if v}}


@app.get("/monitoring", dependencies=[Depends(auth)])
def monitoring():
    """Per test time step: alert rate, score PSI vs train, precision / recall where labels exist."""
    return state()["report"]["monitoring"]
