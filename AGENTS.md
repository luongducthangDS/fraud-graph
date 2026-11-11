# fraud-graph: agent notes

Dự án phủ phần Fraud Detection trong JD (ML, Graph, Anomaly, Semi-supervised, realtime và batch). Commit local, chưa có remote.

Đọc trước khi làm: [docs/RULES.md](docs/RULES.md) > [docs/PRD.md](docs/PRD.md) > [ARCHITECTURE](docs/ARCHITECTURE.md), [DESIGN](docs/DESIGN.md), [TASKS](docs/TASKS.md), [PROGRESS](docs/PROGRESS.md).

## Trạng thái (2026-11-11)
- Xong end to end 09–10/11: batch PR-AUC 0,814 / F1 0,824, realtime PR-AUC 0,810; sau step 43 F1 sập mà PSI không bắt được. Chi tiết ở `docs/PROGRESS.md`.
- Việc tiếp theo: mục đầu tiên chưa tick trong `docs/TASKS.md` (tạo repo GitHub, user tự tạo).
- `python -m fg.train` tự tải dữ liệu và ghi `artifacts/` (~55 phút khi CPU bận).

## Gotchas
- Dùng `.venv` riêng; `python` mặc định của shell là venv DocuMindAI.
- Test hạ `train.N_TREES` xuống 50, đừng bỏ.
- Ngưỡng API 0,24 (OOF) là cố ý; `FG_THRESHOLD` để ghi đè.
- `artifacts/store.parquet` ~75 MB được copy vào Docker image.
