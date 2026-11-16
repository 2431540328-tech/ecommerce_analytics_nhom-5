"""
FILE: main.py (Đặt tại thư mục gốc dự án backend/ hoặc cùng cấp seed_mongodb.py)
MỤC ĐÍCH: Backend FastAPI hoàn chỉnh cho Hệ thống Phân tích & Dự báo Doanh thu TMĐT (FR-01 -> FR-44)
CHẠY SERVER: uvicorn main:app --reload --port 8000
SWAGGER DOCS: http://localhost:8000/docs

DEPENDENCIES CẦN CÀI (pip install):
    fastapi==0.115.0 uvicorn==0.30.6 pymongo==4.8.0 pandas==2.2.2 numpy==1.26.4
    scikit-learn==1.5.1 xgboost==2.1.1 statsmodels==0.14.2 prophet==1.1.6
    pydantic==2.9.0 python-multipart==0.0.9 PyJWT==2.9.0 bcrypt==4.2.0
    joblib==1.4.2 openpyxl==3.1.5
"""

import io
import os
import time
import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

import bcrypt
import joblib
import jwt
import numpy as np
import pandas as pd
import xgboost as xgb
from bson import ObjectId
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from pymongo import ASCENDING, DESCENDING, MongoClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

warnings.filterwarnings("ignore")

# =========================================================================
# CẤU HÌNH HỆ THỐNG & LOGGING (FR-04, FR-40, FR-41)
# =========================================================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "ecommerce_analytics_db")
JWT_SECRET = os.getenv("JWT_SECRET", "vaa-ecommerce-secret-key-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))
MODEL_DIR = os.getenv("MODEL_DIR", "./saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ecommerce_analytics")

app = FastAPI(
    title="API Hệ Thống Phân Tích & Dự Báo Doanh Thu TMĐT",
    description="Admin & Analytics Portal (FastAPI + MongoDB + RFM + SARIMA/Prophet/XGBoost) - FR-01 đến FR-44",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# KẾT NỐI MONGODB & TẠO INDEX (Tối ưu < 500ms)
# =========================================================================
_client: Optional[MongoClient] = None


def get_db():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return _client[DB_NAME]


@app.on_event("startup")
def startup_init():
    """Khởi tạo index MongoDB và cấu hình mặc định khi server khởi động."""
    try:
        db = get_db()
        db.nguoi_dung.create_index([("email", ASCENDING)], unique=True)
        db.san_pham.create_index([("maSKU", ASCENDING)], unique=True)
        db.don_hang.create_index([("ngayDatHang", ASCENDING), ("trangThaiDon", ASCENDING)])
        db.don_hang.create_index([("maDonHang", ASCENDING)], unique=True)
        db.don_hang.create_index([("idKhachHang", ASCENDING)])
        db.khach_hang.create_index([("phanKhucRFM", ASCENDING)])
        db.audit_logs.create_index([("timestamp", DESCENDING)])
        db.system_logs.create_index([("timestamp", DESCENDING)])
        db.du_bao_doanh_thu.create_index([("ngayThucHien", DESCENDING)])

        # Tạo cấu hình hệ thống mặc định nếu chưa có (FR-04)
        if db.system_config.count_documents({"_id": "global"}) == 0:
            db.system_config.insert_one({
                "_id": "global",
                "etl_schedule_cron": "0 2 * * *",
                "alert_forecast_deviation_pct": 15.0,
                "alert_at_risk_spike_pct": 10.0,
                "default_forecast_model": "Hybrid_Prophet_XGBoost",
                "currency": "VND",
                "notify_in_app": True,
                "notify_email": False,
                "updated_at": datetime.now(timezone.utc),
            })

        # Tạo tài khoản mẫu nếu DB trống để test đăng nhập ngay
        if db.nguoi_dung.count_documents({}) == 0:
            pw_hash = bcrypt.hashpw(b"123456", bcrypt.gensalt()).decode("utf-8")
            db.nguoi_dung.insert_many([
                {
                    "hoTen": "Quản Trị Viên",
                    "email": "admin@vaa-ecommerce.vn",
                    "matKhauHash": pw_hash,
                    "vaiTro": "ADMIN",
                    "permissions": ["ALL"],
                    "trangThai": True,
                    "ngayTao": datetime.now(timezone.utc),
                },
                {
                    "hoTen": "Chuyên Viên Dữ Liệu",
                    "email": "analyst@vaa-ecommerce.vn",
                    "matKhauHash": pw_hash,
                    "vaiTro": "DATA_ANALYST",
                    "permissions": ["ETL", "TRAIN_MODEL", "DEEP_ANALYTICS", "MONITORING"],
                    "trangThai": True,
                    "ngayTao": datetime.now(timezone.utc),
                },
            ])
        logger.info("Đã khởi tạo index MongoDB và cấu hình hệ thống thành công.")
    except Exception as exc:
        logger.warning(f"Không thể kết nối MongoDB lúc startup: {exc}")


# =========================================================================
# MIDDLEWARE GIÁM SÁT HIỆU NĂNG & SYSTEM LOG (FR-40, FR-41)
# =========================================================================
@app.middleware("http")
async def performance_and_log_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)

    if request.url.path.startswith("/api/"):
        try:
            db = get_db()
            is_forecast = "/forecast/" in request.url.path or "/models/train" in request.url.path
            sla_limit = 3000.0 if is_forecast else 500.0
            db.system_logs.insert_one({
                "timestamp": datetime.now(timezone.utc),
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": elapsed_ms,
                "sla_breach": elapsed_ms > sla_limit,
            })
        except Exception:
            pass
    return response


# =========================================================================
# TIỆN ÍCH BẢO MẬT JWT, PHÂN QUYỀN & AUDIT LOG (FR-01 -> FR-05)
# =========================================================================
security = HTTPBearer()


def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, list):
            out[k] = [serialize_doc(i) if isinstance(i, dict) else (str(i) if isinstance(i, ObjectId) else i) for i in v]
        elif isinstance(v, dict):
            out[k] = serialize_doc(v)
        else:
            out[k] = v
    return out


def verify_password(plain_pw: str, hashed_pw: str) -> bool:
    try:
        return bcrypt.checkpw(plain_pw.encode("utf-8"), hashed_pw.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user: Dict[str, Any]) -> str:
    payload = {
        "sub": str(user["_id"]),
        "email": user["email"],
        "hoTen": user.get("hoTen", ""),
        "role": user.get("vaiTro", "DATA_ANALYST"),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token đã hết hạn")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token không hợp lệ")

    db = get_db()
    if db.revoked_tokens.find_one({"token": token}):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token đã bị thu hồi")
    return payload


def require_roles(*allowed_roles: str):
    def _checker(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if user.get("role") not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Không đủ quyền thực hiện chức năng này")
        return user
    return _checker


def write_audit_log(user: Dict[str, Any], action: str, target: str, detail: Optional[Dict[str, Any]] = None):
    try:
        get_db().audit_logs.insert_one({
            "timestamp": datetime.now(timezone.utc),
            "user_id": user.get("sub"),
            "email": user.get("email"),
            "role": user.get("role"),
            "action": action,
            "target": target,
            "detail": detail or {},
        })
    except Exception as exc:
        logger.warning(f"Audit log error: {exc}")


# =========================================================================
# PYDANTIC SCHEMAS
# =========================================================================
class LoginRequest(BaseModel):
    email: str
    password: str


class UserCreateRequest(BaseModel):
    hoTen: str
    email: str
    password: str = Field(min_length=6)
    vaiTro: Literal["ADMIN", "DATA_ANALYST"] = "DATA_ANALYST"


class UserUpdateRequest(BaseModel):
    hoTen: Optional[str] = None
    vaiTro: Optional[Literal["ADMIN", "DATA_ANALYST"]] = None
    trangThai: Optional[bool] = None
    new_password: Optional[str] = None
    permissions: Optional[List[str]] = None


class SystemConfigUpdate(BaseModel):
    etl_schedule_cron: Optional[str] = None
    alert_forecast_deviation_pct: Optional[float] = None
    alert_at_risk_spike_pct: Optional[float] = None
    default_forecast_model: Optional[str] = None
    currency: Optional[str] = None
    notify_in_app: Optional[bool] = None
    notify_email: Optional[bool] = None


class TrainModelRequest(BaseModel):
    model_type: Literal["SARIMA", "Prophet", "XGBoost", "Hybrid_Prophet_XGBoost"] = "Hybrid_Prophet_XGBoost"
    category: str = "ALL"
    test_days: int = Field(default=30, ge=7, le=90)
    exclude_anomalies: bool = True
    n_estimators: int = 250
    learning_rate: float = 0.05
    max_depth: int = 5


class ForecastRequest(BaseModel):
    horizon_days: int = Field(default=30, ge=1, le=180)
    category: str = "ALL"
    model_type: Literal["SARIMA", "Prophet", "XGBoost", "Hybrid_Prophet_XGBoost"] = "Hybrid_Prophet_XGBoost"
    confidence_interval: float = Field(default=0.95, ge=0.8, le=0.99)


class CustomAggregateRequest(BaseModel):
    collection: Literal["don_hang", "khach_hang", "san_pham", "du_bao_doanh_thu"]
    pipeline: List[Dict[str, Any]]


class AnomalyCreateRequest(BaseModel):
    start_date: str
    end_date: str
    reason: str
    exclude_from_training: bool = True


class ScheduledReportRequest(BaseModel):
    name: str
    frequency: Literal["DAILY", "WEEKLY", "MONTHLY"]
    recipients: List[str]
    format: Literal["xlsx", "csv"] = "xlsx"
    active: bool = True


# =========================================================================
# 0. PHỤC VỤ GIAO DIỆN WEB (Nếu đặt cùng cấp index.html)
# =========================================================================
@app.get("/", include_in_schema=False)
def serve_frontend():
    for p in ("index.html", "../frontend/index.html"):
        if os.path.exists(p):
            return FileResponse(p)
    return JSONResponse({"message": "Ecommerce Analytics API đang hoạt động. Truy cập /docs để xem API."})


# =========================================================================
# 1. QUẢN TRỊ HỆ THỐNG & NGƯỜI DÙNG (FR-01 -> FR-06)
# =========================================================================
@app.post("/api/v1/auth/login")
def login(req: LoginRequest):
    """FR-01: Đăng nhập & cấp JWT Token."""
    db = get_db()
    user = db.nguoi_dung.find_one({"email": req.email})
    if not user or not verify_password(req.password, user.get("matKhauHash", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Email hoặc mật khẩu không chính xác")
    if not user.get("trangThai", True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tài khoản đang bị khóa")

    token = create_access_token(user)
    write_audit_log({"sub": str(user["_id"]), "email": user["email"], "role": user["vaiTro"]}, "LOGIN", "nguoi_dung")
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": str(user["_id"]),
            "hoTen": user.get("hoTen"),
            "email": user.get("email"),
            "vaiTro": user.get("vaiTro"),
            "permissions": user.get("permissions", []),
        },
    }


@app.post("/api/v1/auth/logout")
def logout(credentials: HTTPAuthorizationCredentials = Depends(security), user: Dict = Depends(get_current_user)):
    """FR-01: Đăng xuất & thu hồi JWT Token."""
    get_db().revoked_tokens.insert_one({"token": credentials.credentials, "revoked_at": datetime.now(timezone.utc)})
    write_audit_log(user, "LOGOUT", "nguoi_dung")
    return {"status": "success", "message": "Đã đăng xuất thành công"}


@app.get("/api/v1/users")
def list_users(user: Dict = Depends(require_roles("ADMIN"))):
    """FR-02: Danh sách tài khoản người dùng."""
    docs = list(get_db().nguoi_dung.find({}, {"matKhauHash": 0}))
    return {"users": [serialize_doc(d) for d in docs]}


@app.post("/api/v1/users", status_code=201)
def create_user(req: UserCreateRequest, user: Dict = Depends(require_roles("ADMIN"))):
    """FR-02: Tạo tài khoản mới."""
    db = get_db()
    if db.nguoi_dung.find_one({"email": req.email}):
        raise HTTPException(status_code=400, detail="Email đã tồn tại")
    pw_hash = bcrypt.hashpw(req.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    doc = {
        "hoTen": req.hoTen,
        "email": req.email,
        "matKhauHash": pw_hash,
        "vaiTro": req.vaiTro,
        "permissions": ["ALL"] if req.vaiTro == "ADMIN" else ["ETL", "TRAIN_MODEL", "DEEP_ANALYTICS", "MONITORING"],
        "trangThai": True,
        "ngayTao": datetime.now(timezone.utc),
    }
    res = db.nguoi_dung.insert_one(doc)
    write_audit_log(user, "CREATE_USER", str(res.inserted_id), {"email": req.email, "role": req.vaiTro})
    return {"id": str(res.inserted_id), "email": req.email, "vaiTro": req.vaiTro}


@app.patch("/api/v1/users/{user_id}")
def update_user(user_id: str, req: UserUpdateRequest, user: Dict = Depends(require_roles("ADMIN"))):
    """FR-02, FR-03: Cập nhật thông tin, khóa/mở khóa, đổi mật khẩu và phân quyền."""
    db = get_db()
    updates: Dict[str, Any] = {}
    if req.hoTen is not None:
        updates["hoTen"] = req.hoTen
    if req.vaiTro is not None:
        updates["vaiTro"] = req.vaiTro
    if req.trangThai is not None:
        updates["trangThai"] = req.trangThai
    if req.permissions is not None:
        updates["permissions"] = req.permissions
    if req.new_password:
        updates["matKhauHash"] = bcrypt.hashpw(req.new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    if not updates:
        raise HTTPException(status_code=400, detail="Không có dữ liệu cập nhật")
    res = db.nguoi_dung.update_one({"_id": ObjectId(user_id)}, {"$set": updates})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")
    write_audit_log(user, "UPDATE_USER", user_id, {k: v for k, v in updates.items() if k != "matKhauHash"})
    return {"status": "updated"}


@app.get("/api/v1/system/config")
def get_system_config(user: Dict = Depends(get_current_user)):
    """FR-04: Lấy cấu hình hệ thống."""
    cfg = get_db().system_config.find_one({"_id": "global"}) or {}
    return serialize_doc(cfg)


@app.put("/api/v1/system/config")
def update_system_config(req: SystemConfigUpdate, user: Dict = Depends(require_roles("ADMIN"))):
    """FR-04, FR-23: Cập nhật cấu hình hệ thống và ngưỡng cảnh báo."""
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    updates["updated_at"] = datetime.now(timezone.utc)
    get_db().system_config.update_one({"_id": "global"}, {"$set": updates}, upsert=True)
    write_audit_log(user, "UPDATE_CONFIG", "system_config", updates)
    return {"status": "updated", "config": serialize_doc(get_db().system_config.find_one({"_id": "global"}))}


@app.get("/api/v1/audit-logs")
def get_audit_logs(limit: int = Query(50, le=500), user: Dict = Depends(require_roles("ADMIN"))):
    """FR-05: Tra cứu nhật ký hoạt động."""
    logs = list(get_db().audit_logs.find().sort("timestamp", DESCENDING).limit(limit))
    return {"logs": [serialize_doc(l) for l in logs]}


# =========================================================================
# 2. DASHBOARD & PHÂN TÍCH DOANH THU ĐA CHIỀU (FR-07 -> FR-12, FR-24)
# =========================================================================
@app.get("/api/v1/analytics/dashboard")
def get_dashboard(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user: Dict = Depends(get_current_user),
):
    """FR-07, FR-08: Executive & Operational Dashboard KPIs (< 500ms)."""
    db = get_db()
    match_stage: Dict[str, Any] = {"trangThaiDon": "COMPLETED"}
    if start_date and end_date:
        match_stage["ngayDatHang"] = {
            "$gte": datetime.fromisoformat(start_date),
            "$lte": datetime.fromisoformat(end_date),
        }

    kpi_res = list(db.don_hang.aggregate([
        {"$match": match_stage},
        {"$group": {
            "_id": None,
            "net_revenue": {"$sum": "$doanhThuThuan"},
            "gmv": {"$sum": "$tongTienHang"},
            "discount_total": {"$sum": "$chietKhau"},
            "total_orders": {"$sum": 1},
            "marketing_spend": {"$sum": "$chiPhiMarketing"},
        }},
    ]))
    kpi = kpi_res[0] if kpi_res else {
        "net_revenue": 0, "gmv": 0, "discount_total": 0, "total_orders": 0, "marketing_spend": 0
    }
    kpi.pop("_id", None)
    kpi["aov"] = round(kpi["net_revenue"] / max(1, kpi["total_orders"]), 0)

    status_breakdown = list(db.don_hang.aggregate([
        {"$group": {"_id": "$trangThaiDon", "count": {"$sum": 1}, "value": {"$sum": "$doanhThuThuan"}}}
    ]))

    monthly_revenue = list(db.don_hang.aggregate([
        {"$match": match_stage},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m", "date": "$ngayDatHang"}},
            "doanhThu": {"$sum": "$doanhThuThuan"},
            "soDon": {"$sum": 1},
        }},
        {"$sort": {"_id": 1}},
    ]))

    rfm_segments = list(db.khach_hang.aggregate([
        {"$group": {"_id": "$phanKhucRFM", "soLuongKhach": {"$sum": 1}, "tongChiTieu": {"$sum": "$chiSoMonetary"}}}
    ]))

    category_breakdown = list(db.don_hang.aggregate([
        {"$match": match_stage},
        {"$unwind": "$danhSachSanPham"},
        {"$group": {
            "_id": "$danhSachSanPham.danhMuc",
            "doanhThu": {"$sum": "$danhSachSanPham.thanhTien"},
            "soLuongBan": {"$sum": "$danhSachSanPham.soLuong"},
        }},
        {"$sort": {"doanhThu": -1}},
    ]))

    return {
        "kpi": kpi,
        "status_breakdown": status_breakdown,
        "monthly_revenue": monthly_revenue,
        "rfm_segments": rfm_segments,
        "category_breakdown": category_breakdown,
    }


@app.get("/api/v1/analytics/multidimensional")
def multidimensional_analytics(
    dimension: Literal["channel", "category", "sku", "hour", "region", "day", "month"] = "channel",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user: Dict = Depends(get_current_user),
):
    """FR-09: Phân tích doanh thu đa chiều theo kênh, danh mục, SKU, khung giờ, khu vực, thời gian."""
    db = get_db()
    match_stage: Dict[str, Any] = {"trangThaiDon": "COMPLETED"}
    if start_date and end_date:
        match_stage["ngayDatHang"] = {"$gte": datetime.fromisoformat(start_date), "$lte": datetime.fromisoformat(end_date)}

    if dimension == "channel":
        pipeline = [
            {"$match": match_stage},
            {"$group": {"_id": "$kenhBanHang", "doanhThu": {"$sum": "$doanhThuThuan"}, "soDon": {"$sum": 1}}},
            {"$sort": {"doanhThu": -1}},
        ]
    elif dimension == "category":
        pipeline = [
            {"$match": match_stage},
            {"$unwind": "$danhSachSanPham"},
            {"$group": {"_id": "$danhSachSanPham.danhMuc", "doanhThu": {"$sum": "$danhSachSanPham.thanhTien"}, "soLuong": {"$sum": "$danhSachSanPham.soLuong"}}},
            {"$sort": {"doanhThu": -1}},
        ]
    elif dimension == "sku":
        pipeline = [
            {"$match": match_stage},
            {"$unwind": "$danhSachSanPham"},
            {"$group": {"_id": "$danhSachSanPham.maSKU", "tenSanPham": {"$first": "$danhSachSanPham.tenSanPham"}, "doanhThu": {"$sum": "$danhSachSanPham.thanhTien"}, "soLuong": {"$sum": "$danhSachSanPham.soLuong"}}},
            {"$sort": {"doanhThu": -1}},
            {"$limit": 20},
        ]
    elif dimension == "hour":
        pipeline = [
            {"$match": match_stage},
            {"$group": {"_id": {"$hour": "$ngayDatHang"}, "doanhThu": {"$sum": "$doanhThuThuan"}, "soDon": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ]
    elif dimension == "region":
        pipeline = [
            {"$match": match_stage},
            {"$lookup": {"from": "khach_hang", "localField": "idKhachHang", "foreignField": "_id", "as": "kh"}},
            {"$unwind": {"path": "$kh", "preserveNullAndEmptyArrays": True}},
            {"$group": {"_id": {"$ifNull": ["$kh.khuVuc", "Khác"]}, "doanhThu": {"$sum": "$doanhThuThuan"}, "soDon": {"$sum": 1}}},
            {"$sort": {"doanhThu": -1}},
        ]
    else:
        fmt = "%Y-%m-%d" if dimension == "day" else "%Y-%m"
        pipeline = [
            {"$match": match_stage},
            {"$group": {"_id": {"$dateToString": {"format": fmt, "date": "$ngayDatHang"}}, "doanhThu": {"$sum": "$doanhThuThuan"}, "soDon": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ]

    return {"dimension": dimension, "data": list(db.don_hang.aggregate(pipeline))}


@app.get("/api/v1/analytics/promotions")
def promotion_effectiveness(user: Dict = Depends(get_current_user)):
    """FR-10: Theo dõi hiệu quả các ngày Double Day / Flash Sale so với ngày thường."""
    db = get_db()
    daily = list(db.don_hang.aggregate([
        {"$match": {"trangThaiDon": "COMPLETED"}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$ngayDatHang"}},
            "doanhThu": {"$sum": "$doanhThuThuan"},
            "chietKhau": {"$sum": "$chietKhau"},
            "soDon": {"$sum": 1},
        }},
        {"$sort": {"_id": 1}},
    ]))
    if not daily:
        return {"promo_days": [], "normal_avg_revenue": 0, "promo_avg_revenue": 0}

    df = pd.DataFrame(daily)
    df["dt"] = pd.to_datetime(df["_id"])
    df["is_promo"] = (df["dt"].dt.day == df["dt"].dt.month) | df["dt"].dt.day.isin([1, 15, 25])
    promo_avg = float(df[df["is_promo"]]["doanhThu"].mean() or 0)
    normal_avg = float(df[~df["is_promo"]]["doanhThu"].mean() or 0)
    uplift_pct = round(((promo_avg - normal_avg) / max(1.0, normal_avg)) * 100, 2)

    return {
        "promo_avg_revenue": round(promo_avg, 0),
        "normal_avg_revenue": round(normal_avg, 0),
        "uplift_pct": uplift_pct,
        "top_promo_days": df[df["is_promo"]].sort_values("doanhThu", ascending=False).head(10)[["_id", "doanhThu", "chietKhau", "soDon"]].to_dict("records"),
    }


@app.get("/api/v1/orders")
def search_orders(
    ma_don: Optional[str] = None,
    status_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(50, le=200),
    user: Dict = Depends(get_current_user),
):
    """FR-24: Tra cứu đơn hàng theo mã đơn, trạng thái, khoảng thời gian."""
    query: Dict[str, Any] = {}
    if ma_don:
        query["maDonHang"] = {"$regex": ma_don, "$options": "i"}
    if status_filter:
        query["trangThaiDon"] = status_filter
    if start_date and end_date:
        query["ngayDatHang"] = {"$gte": datetime.fromisoformat(start_date), "$lte": datetime.fromisoformat(end_date)}

    docs = list(get_db().don_hang.find(query).sort("ngayDatHang", DESCENDING).limit(limit))
    return {"total": len(docs), "orders": [serialize_doc(d) for d in docs]}


# =========================================================================
# 3. PHÂN KHÚC KHÁCH HÀNG RFM & XUẤT DỮ LIỆU (FR-11 -> FR-15)
# =========================================================================
def recalculate_rfm_in_db(db) -> int:
    """Tính toán lại chỉ số R-F-M thực tế từ Collection don_hang."""
    latest = db.don_hang.find_one({"trangThaiDon": "COMPLETED"}, sort=[("ngayDatHang", DESCENDING)])
    ref_date = latest["ngayDatHang"] if latest else datetime.now(timezone.utc)

    agg = list(db.don_hang.aggregate([
        {"$match": {"trangThaiDon": "COMPLETED"}},
        {"$group": {
            "_id": "$idKhachHang",
            "last_date": {"$max": "$ngayDatHang"},
            "freq": {"$sum": 1},
            "monetary": {"$sum": "$doanhThuThuan"},
        }},
    ]))
    if not agg:
        return 0

    df = pd.DataFrame(agg)
    df["recency"] = (ref_date - pd.to_datetime(df["last_date"])).dt.days.clip(lower=0)
    r_q = df["recency"].quantile([0.25, 0.5, 0.75]).to_dict()
    f_q = df["freq"].quantile([0.25, 0.5, 0.75]).to_dict()
    m_q = df["monetary"].quantile([0.25, 0.5, 0.75]).to_dict()

    for _, row in df.iterrows():
        r, f, m = int(row["recency"]), int(row["freq"]), float(row["monetary"])
        if r <= r_q[0.25] and f >= f_q[0.75] and m >= m_q[0.75]:
            seg = "VIP"
        elif f >= f_q[0.5] and r <= r_q[0.5]:
            seg = "Trung thành"
        elif r > r_q[0.75]:
            seg = "Nguy cơ"
        else:
            seg = "Tiềm năng"

        db.khach_hang.update_one(
            {"_id": row["_id"]},
            {"$set": {"chiSoRecency": r, "chiSoFrequency": f, "chiSoMonetary": m, "phanKhucRFM": seg}},
        )
    return len(df)


@app.get("/api/v1/rfm/summary")
def get_rfm_summary(user: Dict = Depends(get_current_user)):
    """FR-13: Báo cáo phân khúc khách hàng RFM."""
    summary = list(get_db().khach_hang.aggregate([
        {"$group": {
            "_id": "$phanKhucRFM",
            "count": {"$sum": 1},
            "avg_recency": {"$avg": "$chiSoRecency"},
            "avg_frequency": {"$avg": "$chiSoFrequency"},
            "total_monetary": {"$sum": "$chiSoMonetary"},
        }},
        {"$sort": {"total_monetary": -1}},
    ]))
    return {"segments": summary}


@app.get("/api/v1/rfm/customers")
def get_rfm_customers(
    segment: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, le=300),
    user: Dict = Depends(get_current_user),
):
    """FR-14: Danh sách khách hàng theo phân khúc RFM."""
    query: Dict[str, Any] = {}
    if segment:
        query["phanKhucRFM"] = segment
    if search:
        query["$or"] = [
            {"tenKhachHang": {"$regex": search, "$options": "i"}},
            {"maKhachHang": {"$regex": search, "$options": "i"}},
        ]
    docs = list(get_db().khach_hang.find(query).sort("chiSoMonetary", DESCENDING).limit(limit))
    return {"customers": [serialize_doc(d) for d in docs]}


@app.get("/api/v1/export/{report_type}")
def export_excel_report(
    report_type: Literal["revenue", "rfm", "forecast", "technical"],
    segment: Optional[str] = None,
    user: Dict = Depends(get_current_user),
):
    """FR-11, FR-15, FR-43: Xuất báo cáo doanh thu / RFM / dự báo / kỹ thuật ra Excel (.xlsx)."""
    db = get_db()
    if report_type == "rfm":
        q = {"phanKhucRFM": segment} if segment else {}
        rows = [serialize_doc(d) for d in db.khach_hang.find(q)]
        df = pd.DataFrame(rows)
        filename = "bao_cao_khach_hang_rfm.xlsx"
    elif report_type == "forecast":
        latest = db.du_bao_doanh_thu.find_one(sort=[("ngayThucHien", DESCENDING)])
        df = pd.DataFrame(latest.get("duLieuDuBao", []) if latest else [])
        filename = "bao_cao_du_bao_doanh_thu.xlsx"
    elif report_type == "technical":
        models = [serialize_doc(m) for m in db.model_registry.find().sort("created_at", DESCENDING)]
        df = pd.DataFrame(models)
        filename = "bao_cao_ky_thuat_ml.xlsx"
    else:
        rows = list(db.don_hang.aggregate([
            {"$match": {"trangThaiDon": "COMPLETED"}},
            {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$ngayDatHang"}}, "doanhThuThuan": {"$sum": "$doanhThuThuan"}, "soDon": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ]))
        df = pd.DataFrame(rows).rename(columns={"_id": "ngay"})
        filename = "bao_cao_doanh_thu.xlsx"

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Report")
    buf.seek(0)
    write_audit_log(user, "EXPORT_REPORT", report_type)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/api/v1/reports/schedule")
def create_scheduled_report(req: ScheduledReportRequest, user: Dict = Depends(require_roles("ADMIN"))):
    """FR-12: Lên lịch gửi báo cáo định kỳ."""
    doc = req.model_dump()
    doc["created_by"] = user.get("email")
    doc["created_at"] = datetime.now(timezone.utc)
    res = get_db().scheduled_reports.insert_one(doc)
    return {"id": str(res.inserted_id), "status": "scheduled"}


# =========================================================================
# 4. PIPELINE HUẤN LUYỆN, ĐÁNH GIÁ & DỰ BÁO ML (FR-06, FR-16->19, FR-25, FR-31->35)
# =========================================================================
def load_daily_revenue_df(db, category: str = "ALL", exclude_anomalies: bool = True) -> pd.DataFrame:
    if category == "ALL":
        pipeline = [
            {"$match": {"trangThaiDon": "COMPLETED"}},
            {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$ngayDatHang"}}, "y": {"$sum": "$doanhThuThuan"}}},
            {"$sort": {"_id": 1}},
        ]
    else:
        pipeline = [
            {"$match": {"trangThaiDon": "COMPLETED"}},
            {"$unwind": "$danhSachSanPham"},
            {"$match": {"danhSachSanPham.danhMuc": category}},
            {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$ngayDatHang"}}, "y": {"$sum": "$danhSachSanPham.thanhTien"}}},
            {"$sort": {"_id": 1}},
        ]
    raw = list(db.don_hang.aggregate(pipeline))
    if len(raw) < 60:
        raise HTTPException(status_code=400, detail="Dữ liệu lịch sử không đủ (cần tối thiểu 60 ngày)")

    df = pd.DataFrame(raw)
    df["ds"] = pd.to_datetime(df["_id"])
    df = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)

    # Loại bỏ các giai đoạn bất thường được Data Analyst đánh dấu (FR-44)
    if exclude_anomalies:
        anomalies = list(db.data_anomalies.find({"exclude_from_training": True}))
        for a in anomalies:
            s_dt, e_dt = pd.to_datetime(a["start_date"]), pd.to_datetime(a["end_date"])
            mask = (df["ds"] >= s_dt) & (df["ds"] <= e_dt)
            if mask.any():
                df.loc[mask, "y"] = df["y"].median()
    return df


def build_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dayofweek"] = out["ds"].dt.dayofweek
    out["day"] = out["ds"].dt.day
    out["month"] = out["ds"].dt.month
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    out["is_double_day"] = ((out["ds"].dt.day == out["ds"].dt.month) | out["ds"].dt.day.isin([1, 15, 25])).astype(int)
    for lag in [1, 7, 14, 30]:
        out[f"lag_{lag}"] = out["y"].shift(lag)
    out["rolling_mean_7"] = out["y"].shift(1).rolling(7, min_periods=1).mean()
    out["rolling_mean_30"] = out["y"].shift(1).rolling(30, min_periods=1).mean()
    out["rolling_std_7"] = out["y"].shift(1).rolling(7, min_periods=1).std().fillna(0)
    return out


FEATURE_COLS = [
    "dayofweek", "day", "month", "is_weekend", "is_double_day",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(1.0, y_true))))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": round(rmse, 1), "MAE": round(mae, 1), "MAPE": round(mape, 4), "R2_Score": round(r2, 4)}


@app.post("/api/v1/models/train")
def train_and_evaluate_model(req: TrainModelRequest, user: Dict = Depends(require_roles("DATA_ANALYST", "ADMIN"))):
    """FR-31, FR-32, FR-33: Huấn luyện, đánh giá trên tập test và lưu phiên bản mô hình."""
    db = get_db()
    df = load_daily_revenue_df(db, req.category, req.exclude_anomalies)
    df_feat = build_xgb_features(df).dropna().reset_index(drop=True)

    train_feat = df_feat.iloc[:-req.test_days]
    test_feat = df_feat.iloc[-req.test_days:]
    y_test = test_feat["y"].values

    xgb_model = xgb.XGBRegressor(
        n_estimators=req.n_estimators,
        learning_rate=req.learning_rate,
        max_depth=req.max_depth,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
    )
    xgb_model.fit(train_feat[FEATURE_COLS], train_feat["y"])
    xgb_preds = xgb_model.predict(test_feat[FEATURE_COLS])

    sarima_res = SARIMAX(
        df["y"].iloc[:-req.test_days],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    sarima_preds = sarima_res.forecast(steps=req.test_days).values

    prophet_preds = None
    if req.model_type in ("Prophet", "Hybrid_Prophet_XGBoost") and PROPHET_AVAILABLE:
        p_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        p_model.fit(df.iloc[:-req.test_days][["ds", "y"]])
        future_p = pd.DataFrame({"ds": test_feat["ds"]})
        prophet_preds = p_model.predict(future_p)["yhat"].values

    if req.model_type == "SARIMA":
        preds_test = sarima_preds
    elif req.model_type == "Prophet":
        preds_test = prophet_preds if prophet_preds is not None else sarima_preds
    elif req.model_type == "XGBoost":
        preds_test = xgb_preds
    else:
        base_trend = prophet_preds if prophet_preds is not None else sarima_preds
        preds_test = 0.7 * xgb_preds + 0.3 * base_trend

    metrics = compute_metrics(y_test, preds_test)

    # Fit lại XGBoost trên toàn bộ dữ liệu để phục vụ inference nhanh < 3s
    full_xgb = xgb.XGBRegressor(
        n_estimators=req.n_estimators,
        learning_rate=req.learning_rate,
        max_depth=req.max_depth,
        random_state=42,
    )
    full_xgb.fit(df_feat[FEATURE_COLS], df_feat["y"])

    version = f"{req.model_type}_{req.category}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    artifact_path = os.path.join(MODEL_DIR, f"{version}.joblib")
    joblib.dump({"xgb": full_xgb, "metrics": metrics}, artifact_path)

    reg_doc = {
        "version": version,
        "model_type": req.model_type,
        "category": req.category,
        "metrics": metrics,
        "params": req.model_dump(),
        "artifact_path": artifact_path,
        "status": "STAGING",  # Chờ Admin phê duyệt lên PRODUCTION (FR-06, FR-34)
        "created_by": user.get("email"),
        "created_at": datetime.now(timezone.utc),
    }
    res = db.model_registry.insert_one(reg_doc)
    write_audit_log(user, "TRAIN_MODEL", version, metrics)

    return {
        "model_id": str(res.inserted_id),
        "version": version,
        "status": "STAGING",
        "metrics": metrics,
        "prophet_installed": PROPHET_AVAILABLE,
    }


@app.get("/api/v1/models")
def list_models(user: Dict = Depends(get_current_user)):
    """FR-06, FR-18, FR-35: Danh sách phiên bản mô hình & bảng so sánh metrics."""
    models = list(get_db().model_registry.find().sort("created_at", DESCENDING))
    return {"models": [serialize_doc(m) for m in models]}


@app.post("/api/v1/models/{model_id}/promote")
def promote_model(model_id: str, user: Dict = Depends(require_roles("ADMIN"))):
    """FR-06, FR-34, FR-35: Phê duyệt / Rollback mô hình lên PRODUCTION."""
    db = get_db()
    target = db.model_registry.find_one({"_id": ObjectId(model_id)})
    if not target:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên bản mô hình")
    db.model_registry.update_many(
        {"category": target["category"], "status": "PRODUCTION"},
        {"$set": {"status": "ARCHIVED"}},
    )
    db.model_registry.update_one({"_id": ObjectId(model_id)}, {"$set": {"status": "PRODUCTION", "approved_by": user.get("email")}})
    write_audit_log(user, "PROMOTE_MODEL", target["version"])
    return {"status": "PRODUCTION", "version": target["version"]}


@app.post("/api/v1/forecast/predict")
def predict_revenue(req: ForecastRequest, user: Dict = Depends(get_current_user)):
    """FR-16, FR-17, FR-19: Dự báo doanh thu ngắn/trung hạn (< 3s) & khuyến nghị hành động."""
    db = get_db()
    df = load_daily_revenue_df(db, req.category, exclude_anomalies=True)
    df_feat = build_xgb_features(df).dropna().reset_index(drop=True)

    # Ưu tiên nạp mô hình PRODUCTION đã train sẵn để đạt inference < 3s
    prod_model = db.model_registry.find_one(
        {"category": req.category, "model_type": req.model_type, "status": "PRODUCTION"},
        sort=[("created_at", DESCENDING)],
    )
    if prod_model and os.path.exists(prod_model["artifact_path"]):
        bundle = joblib.load(prod_model["artifact_path"])
        model_xgb = bundle["xgb"]
        metrics = bundle["metrics"]
    else:
        train_df, test_df = df_feat.iloc[:-30], df_feat.iloc[-30:]
        eval_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        eval_xgb.fit(train_df[FEATURE_COLS], train_df["y"])
        metrics = compute_metrics(test_df["y"].values, eval_xgb.predict(test_df[FEATURE_COLS]))

        model_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        model_xgb.fit(df_feat[FEATURE_COLS], df_feat["y"])

    z_score = 1.96 if req.confidence_interval >= 0.95 else 1.645
    history_df = df.copy()
    last_date = history_df["ds"].max()
    future_records = []

    for step in range(1, req.horizon_days + 1):
        next_date = last_date + timedelta(days=step)
        temp_df = pd.concat([history_df, pd.DataFrame([{"ds": next_date, "y": np.nan}])], ignore_index=True)
        temp_feat = build_xgb_features(temp_df).iloc[-1:]
        pred_val = float(model_xgb.predict(temp_feat[FEATURE_COLS])[0])

        margin = pred_val * (z_score * max(0.04, metrics["MAPE"]))
        pred_round = round(max(0.0, pred_val), 0)
        history_df = pd.concat([history_df, pd.DataFrame([{"ds": next_date, "y": pred_round}])], ignore_index=True)
        future_records.append({
            "ngay": next_date.strftime("%Y-%m-%d"),
            "doanhThuDuBao": pred_round,
            "canDuoi": round(max(0.0, pred_round - margin), 0),
            "canTren": round(pred_round + margin, 0),
        })

    total_forecast = sum(r["doanhThuDuBao"] for r in future_records)
    avg_daily_hist = float(df.tail(30)["y"].mean())
    avg_daily_fore = total_forecast / req.horizon_days
    growth_pct = round(((avg_daily_fore - avg_daily_hist) / max(1.0, avg_daily_hist)) * 100, 2)

    # Khuyến nghị hành động (FR-19)
    recommendations = [
        f"Nhu cầu dự kiến {'tăng' if growth_pct >= 0 else 'giảm'} {abs(growth_pct)}% so với 30 ngày qua; điều chỉnh kế hoạch nhập hàng tương ứng.",
        f"Đề xuất ngân sách Marketing tối ưu (~12% doanh thu dự báo): {round(total_forecast * 0.12):,} VNĐ.",
        "Ưu tiên chiến dịch giữ chân nhóm 'Nguy cơ' và tri ân nhóm 'VIP' trong mô hình RFM.",
    ]

    forecast_doc = {
        "ngayThucHien": datetime.now(timezone.utc),
        "tenMoHinh": req.model_type,
        "chuKyDuBao": req.horizon_days,
        "nganhHang": req.category,
        "chiSoDanhGia": metrics,
        "duLieuDuBao": future_records,
        "khuyenNghi": recommendations,
        "idNguoiTao": user.get("sub"),
        "trangThai": "COMPLETED",
    }
    db.du_bao_doanh_thu.insert_one(forecast_doc)

    recent_history = [
        {"ngay": row["ds"].strftime("%Y-%m-%d"), "doanhThuThucTe": round(float(row["y"]), 0)}
        for _, row in df.tail(60).iterrows()
    ]
    return {
        "status": "COMPLETED",
        "model_used": req.model_type,
        "category": req.category,
        "horizon_days": req.horizon_days,
        "metrics": metrics,
        "historical_series": recent_history,
        "forecast_series": future_records,
        "total_forecast_revenue": total_forecast,
        "recommendations": recommendations,
    }


@app.get("/api/v1/forecast/history")
def get_forecast_history(limit: int = Query(20, le=100), user: Dict = Depends(get_current_user)):
    """FR-25: Xem lịch sử các lần dự báo trước đó."""
    docs = list(get_db().du_bao_doanh_thu.find().sort("ngayThucHien", DESCENDING).limit(limit))
    return {"history": [serialize_doc(d) for d in docs]}


# =========================================================================
# 5. ETL, CHẤT LƯỢNG DỮ LIỆU (IQR), CẢNH BÁO & GIÁM SÁT (FR-20->23, FR-26->30, FR-40->44)
# =========================================================================
@app.post("/api/v1/etl/upload")
async def etl_upload_orders(
    file: UploadFile = File(...),
    user: Dict = Depends(require_roles("DATA_ANALYST", "ADMIN")),
):
    """FR-29, FR-26, FR-27: Import CSV/Excel, làm sạch dữ liệu & ghi log ETL."""
    db = get_db()
    job_id = db.etl_jobs.insert_one({
        "filename": file.filename,
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc),
        "triggered_by": user.get("email"),
    }).inserted_id

    try:
        content = await file.read()
        if file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))

        required_cols = {"maDonHang", "ngayDatHang", "doanhThuThuan"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

        df = df.drop_duplicates(subset=["maDonHang"]).dropna(subset=["maDonHang", "ngayDatHang", "doanhThuThuan"])
        df["ngayDatHang"] = pd.to_datetime(df["ngayDatHang"])

        inserted = 0
        for row in df.to_dict("records"):
            res = db.don_hang.update_one(
                {"maDonHang": str(row["maDonHang"])},
                {"$setOnInsert": {
                    "maDonHang": str(row["maDonHang"]),
                    "ngayDatHang": row["ngayDatHang"].to_pydatetime(),
                    "kenhBanHang": row.get("kenhBanHang", "Website"),
                    "tongTienHang": float(row.get("tongTienHang", row["doanhThuThuan"])),
                    "chietKhau": float(row.get("chietKhau", 0)),
                    "doanhThuThuan": float(row["doanhThuThuan"]),
                    "chiPhiMarketing": float(row.get("chiPhiMarketing", 0)),
                    "trangThaiDon": row.get("trangThaiDon", "COMPLETED"),
                }},
                upsert=True,
            )
            if res.upserted_id:
                inserted += 1

        rfm_updated = recalculate_rfm_in_db(db)
        db.etl_jobs.update_one({"_id": job_id}, {"$set": {"status": "SUCCESS", "finished_at": datetime.now(timezone.utc), "rows_inserted": inserted, "rfm_updated": rfm_updated}})
        write_audit_log(user, "ETL_UPLOAD", file.filename, {"rows_inserted": inserted})
        return {"job_id": str(job_id), "status": "SUCCESS", "rows_inserted": inserted, "rfm_updated": rfm_updated}
    except Exception as exc:
        db.etl_jobs.update_one({"_id": job_id}, {"$set": {"status": "FAILED", "finished_at": datetime.now(timezone.utc), "error": str(exc)}})
        raise HTTPException(status_code=400, detail=f"ETL thất bại: {exc}")


@app.get("/api/v1/etl/jobs")
def list_etl_jobs(limit: int = Query(30, le=100), user: Dict = Depends(require_roles("DATA_ANALYST", "ADMIN"))):
    """FR-26, FR-27: Theo dõi trạng thái & log lỗi ETL."""
    jobs = list(get_db().etl_jobs.find().sort("started_at", DESCENDING).limit(limit))
    return {"jobs": [serialize_doc(j) for j in jobs]}


@app.post("/api/v1/etl/rerun")
def rerun_etl_rfm(user: Dict = Depends(require_roles("DATA_ANALYST", "ADMIN"))):
    """FR-28: Chạy lại đồng bộ tính toán RFM & kiểm tra toàn vẹn dữ liệu."""
    updated = recalculate_rfm_in_db(get_db())
    return {"status": "SUCCESS", "rfm_recalculated": updated}


@app.get("/api/v1/data-quality/report")
def data_quality_report(user: Dict = Depends(require_roles("DATA_ANALYST", "ADMIN"))):
    """FR-30, FR-42: Kiểm tra chất lượng dữ liệu & phát hiện ngoại lai bằng IQR."""
    db = get_db()
    revenues = [d["doanhThuThuan"] for d in db.don_hang.find({"trangThaiDon": "COMPLETED"}, {"doanhThuThuan": 1})]
    if not revenues:
        return {"total_orders": 0, "outliers_iqr": 0}

    s = pd.Series(revenues)
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers_count = int(((s < lower) | (s > upper)).sum())
    missing_customer = db.don_hang.count_documents({"idKhachHang": {"$exists": False}})

    return {
        "total_completed_orders": len(revenues),
        "iqr_bounds": {"lower": round(lower, 0), "upper": round(upper, 0)},
        "outliers_iqr_count": outliers_count,
        "missing_customer_ref": missing_customer,
    }


@app.get("/api/v1/alerts")
def get_active_alerts(user: Dict = Depends(get_current_user)):
    """FR-20, FR-21, FR-22: Cảnh báo lệch dự báo, tồn kho rủi ro, khách hàng rời bỏ."""
    db = get_db()
    cfg = db.system_config.find_one({"_id": "global"}) or {}
    dev_limit = cfg.get("alert_forecast_deviation_pct", 15.0)
    alerts = []

    # Kiểm tra tồn kho thấp (FR-21)
    low_stock = list(db.san_pham.find({"tonKho": {"$lt": 50}}, {"maSKU": 1, "tenSanPham": 1, "tonKho": 1}))
    for sp in low_stock:
        alerts.append({"type": "INVENTORY_RISK", "severity": "HIGH", "message": f"SKU {sp['maSKU']} ({sp['tenSanPham']}) tồn kho thấp: {sp['tonKho']}"})

    # Kiểm tra tỷ lệ khách hàng Nguy cơ (FR-22)
    total_kh = max(1, db.khach_hang.count_documents({}))
    at_risk_kh = db.khach_hang.count_documents({"phanKhucRFM": "Nguy cơ"})
    at_risk_pct = round((at_risk_kh / total_kh) * 100, 1)
    if at_risk_pct >= cfg.get("alert_at_risk_spike_pct", 10.0):
        alerts.append({"type": "CHURN_RISK", "severity": "MEDIUM", "message": f"Nhóm khách hàng 'Nguy cơ' chiếm {at_risk_pct}% ({at_risk_kh}/{total_kh})"})

    # Kiểm tra lệch dự báo gần nhất (FR-20)
    latest_fc = db.du_bao_doanh_thu.find_one(sort=[("ngayThucHien", DESCENDING)])
    if latest_fc and latest_fc.get("chiSoDanhGia", {}).get("MAPE", 0) * 100 > dev_limit:
        alerts.append({"type": "FORECAST_DEVIATION", "severity": "HIGH", "message": f"MAPE mô hình vượt ngưỡng {dev_limit}%."})

    return {"alerts": alerts}


# =========================================================================
# 6. PHÂN TÍCH CHUYÊN SÂU & GIÁM SÁT (FR-36 -> FR-41, FR-44)
# =========================================================================
@app.post("/api/v1/analytics/custom-aggregate")
def custom_aggregate(req: CustomAggregateRequest, user: Dict = Depends(require_roles("DATA_ANALYST", "ADMIN"))):
    """FR-36: Truy vấn Aggregation Pipeline tùy chỉnh (chỉ cho phép đọc)."""
    forbidden_stages = {"$out", "$merge"}
    for stage in req.pipeline:
        if any(k in forbidden_stages for k in stage.keys()):
            raise HTTPException(status_code=400, detail="Không cho phép stage ghi dữ liệu ($out, $merge)")
    res = list(get_db()[req.collection].aggregate(req.pipeline))
    return {"count": len(res), "results": [serialize_doc(r) if isinstance(r, dict) else r for r in res]}


@app.get("/api/v1/analytics/cohort")
def cohort_analysis(user: Dict = Depends(require_roles("DATA_ANALYST", "ADMIN"))):
    """FR-37, FR-38: Phân tích Cohort khách hàng & CLV trung bình."""
    orders = list(get_db().don_hang.find({"trangThaiDon": "COMPLETED"}, {"idKhachHang": 1, "ngayDatHang": 1, "doanhThuThuan": 1}))
    if not orders:
        return {"cohorts": [], "avg_clv": 0}

    df = pd.DataFrame(orders)
    df["order_month"] = pd.to_datetime(df["ngayDatHang"]).dt.to_period("M")
    df["cohort_month"] = df.groupby("idKhachHang")["order_month"].transform("min")
    df["period_number"] = (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)

    cohort_pivot = df.groupby(["cohort_month", "period_number"])["idKhachHang"].nunique().unstack(fill_value=0)
    cohort_dict = {str(idx): row.to_dict() for idx, row in cohort_pivot.tail(12).iterrows()}
    avg_clv = round(float(df.groupby("idKhachHang")["doanhThuThuan"].sum().mean()), 0)
    return {"cohorts": cohort_dict, "avg_clv": avg_clv}


@app.get("/api/v1/analytics/decomposition")
def time_series_decomposition(category: str = "ALL", user: Dict = Depends(require_roles("DATA_ANALYST", "ADMIN"))):
    """FR-39: Phân rã chuỗi thời gian (Trend, Seasonality, Residual)."""
    df = load_daily_revenue_df(get_db(), category, exclude_anomalies=False).set_index("ds").asfreq("D").fillna(method="ffill")
    decomp = seasonal_decompose(df["y"], model="additive", period=7)
    tail_idx = df.tail(60).index
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in tail_idx],
        "trend": [round(float(v), 0) if pd.notna(v) else None for v in decomp.trend.tail(60)],
        "seasonal": [round(float(v), 0) if pd.notna(v) else None for v in decomp.seasonal.tail(60)],
        "residual": [round(float(v), 0) if pd.notna(v) else None for v in decomp.resid.tail(60)],
    }


@app.post("/api/v1/data-quality/anomalies")
def mark_anomaly(req: AnomalyCreateRequest, user: Dict = Depends(require_roles("DATA_ANALYST", "ADMIN"))):
    """FR-44: Đánh dấu giai đoạn dữ liệu bất thường để loại trừ khi huấn luyện."""
    doc = req.model_dump()
    doc["created_by"] = user.get("email")
    doc["created_at"] = datetime.now(timezone.utc)
    res = get_db().data_anomalies.insert_one(doc)
    return {"id": str(res.inserted_id), "status": "marked"}


@app.get("/api/v1/monitoring/performance")
def get_performance_metrics(user: Dict = Depends(require_roles("DATA_ANALYST", "ADMIN"))):
    """FR-40, FR-41: Giám sát hiệu năng SLA (< 500ms truy vấn, < 3s inference) & log hệ thống."""
    db = get_db()
    recent_logs = list(db.system_logs.find().sort("timestamp", DESCENDING).limit(100))
    durations = [l["duration_ms"] for l in recent_logs]
    breaches = sum(1 for l in recent_logs if l.get("sla_breach"))
    return {
        "avg_latency_ms": round(sum(durations) / max(1, len(durations)), 2),
        "max_latency_ms": max(durations, default=0),
        "sla_breaches_in_last_100": breaches,
        "recent_logs": [serialize_doc(l) for l in recent_logs[:20]],
    }
