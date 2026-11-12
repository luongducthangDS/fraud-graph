# fraud-graph: agent notes

Dự án phủ phần Fraud Detection trong JD (ML, Graph, Anomaly, Semi-supervised, realtime và batch). Public tại github.com/luongducthangDS/fraud-graph.

11/11: thư mục mất khi di chuyển, khôi phục từ lịch sử Claude Code; 12/11 đã push và train lại `artifacts/` (số khớp).

Đọc trước khi làm: [docs/RULES.md](docs/RULES.md) > [docs/PRD.md](docs/PRD.md) > [ARCHITECTURE](docs/ARCHITECTURE.md), [DESIGN](docs/DESIGN.md), [TASKS](docs/TASKS.md), [PROGRESS](docs/PROGRESS.md).

## Trạng thái (2026-11-12)
- Batch PR-AUC 0,814 / F1 0,824, realtime PR-AUC 0,810; sau step 43 F1 sập mà PSI không bắt được. Chi tiết ở `docs/PROGRESS.md`.
- 12/11: regression gate trong CI, API key, log JSON, `/metrics`, `action: analyst_review`, sơ đồ Mermaid.
- Việc tiếp theo: mục đầu tiên chưa tick trong `docs/TASKS.md`.

## Gotchas
- Dùng `.venv` riêng; `python` mặc định của shell là venv DocuMindAI.
- Test hạ `train.N_TREES` xuống 50, đừng bỏ.
- Ngưỡng API 0,24 (OOF) là cố ý; `FG_THRESHOLD` để ghi đè.
- `artifacts/store.parquet` (~36 MB, gitignored) được copy vào Docker image; chỉ `report.json` được commit.
- Gate đọc `artifacts/report.json`: train lại thì chạy `python -m fg.gate`; cập nhật `eval/baseline.json` chỉ khi có chủ đích.
