"""
FILE: seed_mongodb.py
Mục đích: Tạo tự động 5 Collection trong MongoDB (ecommerce_analytics_db) khớp 100% với Báo cáo:
1. nguoi_dung (Users)
2. san_pham (Products)
3. khach_hang (Customers - kèm tính toán chỉ số RFM)
4. don_hang (Orders - dữ liệu chuỗi thời gian 24 tháng có mùa vụ & Double Days 9/9, 11/11, 12/12)
5. du_bao_doanh_thu (ForecastResults)
"""

import random
from datetime import datetime, timedelta
import numpy as np
import bcrypt
from pymongo import MongoClient, ASCENDING

# Kết nối MongoDB Local (mặc định cổng 27017 trên MongoDB Compass)
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "ecommerce_analytics_db"

def seed_database():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print(f"[*] Đang khởi tạo cơ sở dữ liệu: {DB_NAME} ...")

    # Xóa dữ liệu cũ nếu chạy lại
    db.nguoi_dung.drop()
    db.san_pham.drop()
    db.khach_hang.drop()
    db.don_hang.drop()
    db.du_bao_doanh_thu.drop()

    # =========================================================================
    # 1. COLLECTION: nguoi_dung (Bảng 3.6 trong Báo cáo)
    # =========================================================================
    users = [
        {
            "hoTen": "Nguyễn Văn A (Nhóm 01)",
            "email": "admin@vaa-ecommerce.vn",
            "matKhauHash": bcrypt.hashpw(b"123456", bcrypt.gensalt()).decode("utf-8"),
            "vaiTro": "ADMIN",
            "trangThai": True,
            "ngayTao": datetime(2025, 1, 1, 8, 0, 0)
        },
        {
            "hoTen": "Trần Thị B",
            "email": "analyst@vaa-ecommerce.vn",
            "matKhauHash": bcrypt.hashpw(b"123456", bcrypt.gensalt()).decode("utf-8"),
            "vaiTro": "DATA_ANALYST",
            "trangThai": True,
            "ngayTao": datetime(2025, 1, 5, 9, 30, 0)
        }
    ]
    user_res = db.nguoi_dung.insert_many(users)
    db.nguoi_dung.create_index([("email", ASCENDING)], unique=True)
    admin_id = user_res.inserted_ids[0]
    print(f"[+] Đã tạo Collection 'nguoi_dung': {len(users)} bản ghi.")

    # =========================================================================
    # 2. COLLECTION: san_pham (Bảng 3.7 trong Báo cáo)
    # =========================================================================
    categories = [
        ("Thoi_Trang_Nam_Nu", "Áo Polo Coolmax Cao Cấp", 250000, 140000),
        ("Thoi_Trang_Nam_Nu", "Quần Jeans Slimfit Hàn Quốc", 420000, 260000),
        ("Thoi_Trang_Nam_Nu", "Áo Khoác Bomber Chống Nước", 550000, 320000),
        ("Phu_Kien_Cong_Nghe", "Tai Nghe Bluetooth Chống Ồn ANC", 680000, 410000),
        ("Phu_Kien_Cong_Nghe", "Sạc Dự Phòng FastCharge 20000mAh", 450000, 270000),
        ("Phu_Kien_Cong_Nghe", "Chuột Không Dây Công Thái Học", 320000, 180000),
        ("My_Pham_Cham_Soc", "Serum Vitamin C Sáng Da 30ml", 380000, 190000),
        ("My_Pham_Cham_Soc", "Kem Chống Nắng Phổ Rộng SPF50+", 290000, 150000),
        ("Gia_Dung_Thong_Minh", "Máy Xay Sinh Tố Cầm Tay Mini", 350000, 210000),
        ("Gia_Dung_Thong_Minh", "Bình Giữ Nhiệt Inox 304 Hiển Thị Nhiệt Độ", 190000, 95000)
    ]
    products = []
    for idx, (cat, name, price, cost) in enumerate(categories, start=1):
        products.append({
            "maSKU": f"SKU-VAA-{idx:03d}",
            "tenSanPham": name,
            "danhMuc": cat,
            "giaBan": price,
            "giaVon": cost,
            "tonKho": random.randint(120, 850)
        })
    prod_res = db.san_pham.insert_many(products)
    db.san_pham.create_index([("maSKU", ASCENDING)], unique=True)
    product_docs = list(db.san_pham.find())
    print(f"[+] Đã tạo Collection 'san_pham': {len(products)} sản phẩm.")

    # =========================================================================
    # 3. COLLECTION: khach_hang (Bảng 3.8 trong Báo cáo)
    # =========================================================================
    provinces = ["TP. Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Cần Thơ", "Bình Dương", "Đồng Nai", "Hải Phòng"]
    first_names = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Đặng", "Bùi"]
    mid_names = ["Văn", "Thị", "Hoàng", "Minh", "Ngọc", "Thanh", "Đức", "Bảo", "Gia", "Tuấn"]
    last_names = ["An", "Bình", "Châu", "Dũng", "Em", "Giang", "Hương", "Khánh", "Linh", "Nam", "Phúc", "Quân", "Tâm", "Vy"]

    customers = []
    for i in range(1, 301):
        full_name = f"{random.choice(first_names)} {random.choice(mid_names)} {random.choice(last_names)}"
        customers.append({
            "maKhachHang": f"KH{i:04d}",
            "tenKhachHang": full_name,
            "khuVuc": random.choices(provinces, weights=[40, 25, 10, 5, 10, 5, 5])[0],
            "chiSoRecency": 999,
            "chiSoFrequency": 0,
            "chiSoMonetary": 0,
            "phanKhucRFM": "Tiềm năng"
        })
    cust_res = db.khach_hang.insert_many(customers)
    customer_ids = cust_res.inserted_ids
    print(f"[+] Đã tạo Collection 'khach_hang': {len(customers)} khách hàng.")

    # =========================================================================
    # 4. COLLECTION: don_hang (Bảng 3.9 trong Báo cáo)
    # Tạo chuỗi thời gian từ 01/01/2025 đến 31/10/2026 (669 ngày)
    # =========================================================================
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 10, 31)
    total_days = (end_date - start_date).days + 1

    channels = ["Shopee", "TikTok Shop", "Website", "Lazada"]
    orders_batch = []
    cust_stats = {cid: {"last_date": start_date, "freq": 0, "monetary": 0} for cid in customer_ids}

    order_seq = 10000
    np.random.seed(42)
    random.seed(42)

    for day_offset in range(total_days):
        curr_date = start_date + timedelta(days=day_offset)
        
        # Xu hướng tăng trưởng theo thời gian + Mùa vụ cuối tuần + Sự kiện Double Day (9/9, 10/10, 11/11, 12/12)
        trend_factor = 1.0 + (day_offset / total_days) * 0.65
        weekend_boost = 1.25 if curr_date.weekday() in [5, 6] else 1.0
        double_day_boost = 2.1 if (curr_date.day == curr_date.month) or (curr_date.day in [1, 15, 25]) else 1.0
        year_end_boost = 1.35 if curr_date.month in [10, 11, 12, 1] else 1.0

        expected_orders = int(14 * trend_factor * weekend_boost * double_day_boost * year_end_boost + np.random.normal(0, 2))
        expected_orders = max(5, expected_orders)

        daily_mkt_cost = round(expected_orders * random.uniform(22000, 35000), -3)

        for _ in range(expected_orders):
            order_seq += 1
            cid = random.choice(customer_ids)
            num_items = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
            chosen_prods = random.sample(product_docs, num_items)

            items_list = []
            gmv = 0
            for p in chosen_prods:
                qty = random.choices([1, 2, 3], weights=[75, 20, 5])[0]
                line_total = qty * p["giaBan"]
                gmv += line_total
                items_list.append({
                    "idSanPham": p["_id"],
                    "maSKU": p["maSKU"],
                    "tenSanPham": p["tenSanPham"],
                    "danhMuc": p["danhMuc"],
                    "soLuong": qty,
                    "donGia": p["giaBan"],
                    "thanhTien": line_total
                })

            discount = round(gmv * random.choice([0, 0.05, 0.1, 0.15]), -3)
            net_rev = gmv - discount
            status = random.choices(["COMPLETED", "CANCELLED", "RETURNED"], weights=[92, 5, 3])[0]

            orders_batch.append({
                "maDonHang": f"ORD-{curr_date.strftime('%Y%m')}-{order_seq}",
                "idKhachHang": cid,
                "ngayDatHang": curr_date + timedelta(hours=random.randint(7, 23), minutes=random.randint(0, 59)),
                "kenhBanHang": random.choices(channels, weights=[45, 30, 15, 10])[0],
                "danhSachSanPham": items_list,
                "tongTienHang": gmv,
                "chietKhau": discount,
                "doanhThuThuan": net_rev,
                "chiPhiMarketing": round(daily_mkt_cost / expected_orders, 0),
                "trangThaiDon": status
            })

            if status == "COMPLETED":
                cust_stats[cid]["last_date"] = max(cust_stats[cid]["last_date"], curr_date)
                cust_stats[cid]["freq"] += 1
                cust_stats[cid]["monetary"] += net_rev

    db.don_hang.insert_many(orders_batch)
    db.don_hang.create_index([("ngayDatHang", ASCENDING)])
    db.don_hang.create_index([("maDonHang", ASCENDING)], unique=True)
    print(f"[+] Đã tạo Collection 'don_hang': {len(orders_batch)} đơn hàng lịch sử.")

    # Cập nhật chỉ số RFM thực tế cho từng Khách hàng
    ref_date = end_date
    for cid, st in cust_stats.items():
        r_days = (ref_date - st["last_date"]).days
        freq = st["freq"]
        mon = st["monetary"]

        if freq >= 45 and mon >= 22000000 and r_days <= 25:
            segment = "VIP"
        elif freq >= 28 and r_days <= 45:
            segment = "Trung thành"
        elif r_days > 60 or freq <= 12:
            segment = "Nguy cơ"
        else:
            segment = "Tiềm năng"

        db.khach_hang.update_one(
            {"_id": cid},
            {"$set": {
                "chiSoRecency": r_days,
                "chiSoFrequency": freq,
                "chiSoMonetary": mon,
                "phanKhucRFM": segment
            }}
        )
    print("[+] Đã tính toán và cập nhật phân khúc RFM cho toàn bộ 'khach_hang'.")

    # =========================================================================
    # 5. COLLECTION: du_bao_doanh_thu (Bảng 3.10 trong Báo cáo)
    # =========================================================================
    sample_forecast = {
        "ngayThucHien": datetime(2026, 11, 1, 8, 30, 0),
        "tenMoHinh": "XGBoost_Prophet_Ensemble_v2",
        "chuKyDuBao": 30,
        "nganhHang": "ALL",
        "chiSoDanhGia": {
            "RMSE": 14250000.5,
            "MAE": 10820000.0,
            "MAPE": 0.0482,
            "R2_Score": 0.941
        },
        "duLieuDuBao": [
            {"ngay": "2026-11-01", "doanhThuDuBao": 21500000, "canDuoi": 19800000, "canTren": 23200000},
            {"ngay": "2026-11-02", "doanhThuDuBao": 22850000, "canDuoi": 21000000, "canTren": 24700000}
        ],
        "idNguoiTao": admin_id,
        "trangThai": "COMPLETED"
    }
    db.du_bao_doanh_thu.insert_one(sample_forecast)
    print("[+] Đã tạo Collection 'du_bao_doanh_thu' thành công!")
    print("\n=> HOÀN TẤT! Mở MongoDB Compass kết nối 'mongodb://localhost:27017' để xem 5 Collection.")

if __name__ == "__main__":
    seed_database()
