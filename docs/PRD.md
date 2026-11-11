# PRD: Fraud Graph

## Bài toán
Phát hiện giao dịch bất hợp pháp trên đồ thị thanh toán Bitcoin (Elliptic: 203.769 giao dịch, 234.355 cạnh, 49 time step, 77 % chưa có nhãn), cả **realtime** (khi giao dịch đến) và **batch** (cuối step).

## Người dùng
- Đội phòng chống gian lận / AML: điểm rủi ro, cảnh báo, theo dõi drift.
- Người đọc portfolio: dự án phủ phần Fraud Detection trong JD (ML, Graph, Anomaly, Semi-supervised). (unverified: suy ra từ AGENTS.md)

## Mục tiêu
1. So sánh trung thực: anomaly không giám sát, LightGBM, collective classification hai tầng, Random Forest, self-training.
2. Realtime chỉ dùng thông tin có tại thời điểm giao dịch (cạnh vào); batch dùng cả cạnh ra. Không train/serve skew.
3. API realtime có parity với offline, batch job, monitoring drift, Docker, CI.

## Không làm
- GNN (chưa thử).
- Online store phân tán (đang trong bộ nhớ, một tiến trình).
- Vòng phản hồi nhãn thật cho monitoring recall.
