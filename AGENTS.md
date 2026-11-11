# fraud-graph: agent notes

Dự án phủ phần Fraud Detection trong JD (ML, Graph, Anomaly, Semi-supervised, realtime và batch). Chưa có remote.

**11/11: thư mục bị mất khi di chuyển (robocopy) rồi được khôi phục từ lịch sử Claude Code** (commit mới `89c24f0`; các commit cũ `4b83ed1`, `fa8708f` không còn). Code, docs, test (5/5 pass), CI, Dockerfile đã về. Mất: `artifacts/` (gồm `report.json`) và `data/`; chạy lại `python -m fg.train` để tạo lại. Việc đầu tiên: tạo repo GitHub và push để không mất lần nữa.

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
