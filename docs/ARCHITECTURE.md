# Architecture

## Stack
Python 3.11 · pandas/pyarrow · scipy sparse (adjacency) · LightGBM · scikit-learn (Isolation Forest, Random Forest) · FastAPI · pytest · Docker · GitHub Actions.

## Thư mục
```
fg/data.py     tải mirror HF yhoma/elliptic-bitcoin-dataset → data/raw/, cache data/nodes.parquet
fg/graph.py    adjacency trong cùng step, đặc trưng hàng xóm vào/ra, mean/max điểm hàng xóm
fg/train.py    backtest (train step 1–34, test 35–49), OOF theo khối thời gian, hai tầng realtime/batch,
               IF, RF, self-training, PSI monitoring → artifacts/ (report.json, store.parquet ~75 MB)
fg/api.py      POST /score realtime, GET /monitoring, GET /health; online store trong bộ nhớ
fg/batch.py    chấm batch cuối step (cạnh vào + ra)
tests/test_pipeline.py  không dùng nhãn test, parity API/offline, ... (N_TREES=50)
```

## Luồng dữ liệu
```
giao dịch mới + input ids → online store (đặc trưng local + điểm tầng 1 của giao dịch đã biết)
  → LightGBM tầng 1 (AF + NE_in) → s1 → ghi lại store
  → LightGBM tầng 2 (AF + NE_in + S_in: mean/max s1 hàng xóm) → điểm, cảnh báo (ngưỡng 0,24)
cuối step: batch.py dùng thêm cạnh ra (NE_out, S_out)
```
