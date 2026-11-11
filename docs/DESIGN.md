# Design (không có UI: định dạng đầu ra)

## API (JSON)
- `POST /score`: nhận giao dịch mới (đặc trưng local + danh sách input ids), trả điểm tầng 1, điểm cuối, cờ cảnh báo theo ngưỡng (mặc định 0,24, ghi đè bằng `FG_THRESHOLD`).
- `GET /monitoring`: số request, latency, PSI phân phối điểm.
- `GET /health`.

## Báo cáo
`artifacts/report.json` là nguồn số; README trình bày bảng tiếng Việt (số thập phân dấu phẩy). Precision/recall/F1 ghi rõ ngưỡng (0,5 trong bảng), PR-AUC không phụ thuộc ngưỡng. Kết quả xấu (Isolation Forest, sập sau step 43) được ghi thẳng.
