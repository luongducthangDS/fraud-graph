# Progress

## Trạng thái (2026-11-12)
Xong end to end, public tại github.com/luongducthangDS/fraud-graph, CI xanh.

Kết quả test (step 35–49, `artifacts/report.json`, train lại 12/11 ra đúng số 10/11):
- Batch tầng 2: PR-AUC 0,814, F1 0,824. Realtime: PR-AUC 0,810. Random Forest: F1 0,813.
- Isolation Forest: PR-AUC 0,036 (vô dụng trên bộ này).
- Sau step 43 F1 rơi xuống 0,03 mà PSI không phát hiện; self-training không giúp.

Đã kiểm: `pytest` 8 passed; `python -m fg.gate` PASS; parity API/offline trên 200 giao dịch thật, sai lệch tối đa 1,3e-4; latency p50 23 ms. `python -m fg.train` 1.289 s (12/11). `docker run` với `FG_API_KEY`: thiếu key 401, `/score` trả `action` + `model_version`, `/metrics` có tỉ lệ cảnh báo; image 1,07 GB.

## Điểm yếu đã biết
- Tỉ lệ cảnh báo trong `/metrics` là tín hiệu thô, chưa có ngưỡng báo động; chưa kiểm nó bắt được sự cố step 43 khi replay.
- Online store và metrics trong bộ nhớ một tiến trình; Docker image nặng vì copy `store.parquet` ~36 MB + LightGBM.

## Bước tiếp theo
Mục đầu tiên chưa tick trong `TASKS.md`.
