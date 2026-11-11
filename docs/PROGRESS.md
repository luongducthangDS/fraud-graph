# Progress

## Trạng thái (2026-11-10)
Xong end to end, commit local, **chưa có remote**.

Kết quả test (step 35–49, `artifacts/report.json`):
- Batch tầng 2: PR-AUC 0,814, F1 0,824. Realtime: PR-AUC 0,810. Random Forest: F1 0,813.
- Isolation Forest: PR-AUC 0,036 (vô dụng trên bộ này).
- Sau step 43 F1 rơi xuống 0,03 mà PSI không phát hiện; self-training không giúp.

Đã kiểm: `pytest` 5 passed (~2,5 phút); parity API/offline trên 200 giao dịch thật, sai lệch tối đa 1,3e-4; latency p50 23 ms. Một lần `python -m fg.train` đầy đủ mất 3.259 s (khi CPU chia với PhoBERT).

## Điểm yếu đã biết
- Monitoring hiện tại (PSI điểm) không bắt được sự cố sau step 43.
- Online store trong bộ nhớ; Docker image nặng vì copy `store.parquet` ~75 MB.

## Bước tiếp theo
Mục đầu tiên chưa tick trong `TASKS.md`: tạo repo GitHub.
