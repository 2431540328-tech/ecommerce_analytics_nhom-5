# Hệ thống Phân tích, Báo cáo & Dự báo Doanh thu TMĐT

## Cấu trúc thư mục
- `backend/main.py`: FastAPI + JWT RBAC + MongoDB Aggregation + RFM + SARIMA/Prophet/XGBoost + ETL + Monitoring (FR-01 → FR-44)
- `backend/seed_mongodb.py`: Khởi tạo dữ liệu mẫu 5 collection vào MongoDB
- `backend/requirements.txt`: Thư viện Python cần cài
- `frontend/index.html`: Giao diện React 18 + TailwindCSS + Chart.js

## Hướng dẫn chạy
1. Cài đặt thư viện:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. Khởi tạo dữ liệu MongoDB (đảm bảo MongoDB đang chạy tại `localhost:27017`):
   ```bash
   python seed_mongodb.py
   ```
3. Khởi chạy API & Web Portal:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
4. Truy cập trình duyệt:
   - Giao diện Web: `http://localhost:8000/`
   - Tài liệu API (Swagger): `http://localhost:8000/docs`
   - Tài khoản mặc định (khi DB chưa có user hoặc dùng user startup): `admin@vaa-ecommerce.vn` / `analyst@vaa-ecommerce.vn` (mật khẩu: `123456`)
