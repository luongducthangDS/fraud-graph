# Tasks

## Đã xong (2026-11-09 → 10)
- [x] Dữ liệu, đặc trưng đồ thị, hai tầng realtime/batch, baseline IF/RF, self-training
- [x] API realtime (parity), batch job, PSI monitoring, Docker, CI

## Tiếp theo (lần lượt từ trên xuống)
- [x] (12/11) Tạo repo GitHub và push; train lại artifacts (số khớp README), commit `report.json`
- [x] (12/11) Chuẩn CV: regression gate `fg/gate.py` + `eval/baseline.json` trong CI; API key, validate input, 503, `action: analyst_review`, log JSON + request id + model version, `/metrics` có tỉ lệ cảnh báo gần đây; sơ đồ Mermaid + quyết định thiết kế
- [ ] Monitoring bắt được sập sau step 43: theo dõi tỉ lệ cảnh báo / phân phối đặc trưng chính, không chỉ PSI điểm
- [ ] Online store dùng Redis thay bộ nhớ một tiến trình
- [ ] Thử GNN (GraphSAGE) so với hai tầng LightGBM
