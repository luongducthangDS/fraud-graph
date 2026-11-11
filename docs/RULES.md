# Rules

## Bất biến (không gỡ)
- Chia theo thời gian: train step 1–34, test 35–49. Test `test_test_labels_never_used` (lật nhãn test, điểm không đổi) phải luôn pass.
- Mô hình realtime chỉ dùng cạnh vào; cạnh ra chỉ cho batch. Thêm đặc trưng realtime nào cũng phải kiểm parity API so với offline.
- Ngưỡng API 0,24 chọn trên out-of-fold. Không đổi sang 0,5 dù test đẹp hơn (nhìn test là rò rỉ); dùng `FG_THRESHOLD` nếu cần ghi đè khi triển khai.
- Test hạ `train.N_TREES` xuống 50; không bỏ, nếu không CI mất hơn 10 phút.

## Quy ước
- Dùng `.venv` riêng; `python` mặc định của shell là venv DocuMindAI.
- LightGBM `n_jobs=1` khi máy có job nặng khác; máy rảnh có thể tăng.
- Số trong README lấy từ `artifacts/report.json`.

## AI không bao giờ được
- Commit `data/` hoặc `artifacts/store.parquet`.
- Tạo remote/push khi user chưa yêu cầu.
