import os
import logging
import numpy as np
import onnxruntime as ort
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, status, Depends
from pydantic import BaseModel, Field
from scipy.special import softmax
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, SmallInteger, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 1. 读取云托管自动注入的环境变量 ---

DB_HOST = os.getenv("DB_HOST", "sh-cynosdbmysql-grp-qj85u8vm.sql.tencentcdb.com")
DB_PORT = os.getenv("DB_PORT", "25802")
DB_USER = os.getenv("DB_USER", "jack")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")  # 读取到正确密码
DB_NAME = os.getenv("DB_NAME", "cloud1-d8g5he955c1d2cf68")

# 拼接 SQLAlchemy 数据库 URL
DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# 配置 DB 引擎（连接超时设置为 5 秒，防卡死）
engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True, 
    pool_recycle=3600,
    connect_args={"connect_timeout": 5}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- 2. ORM 数据表模型定义 ---

# 2.1 预测历史记录表
class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    _openid = Column(String(64), index=True)
    cfbg = Column(Float)
    cdbp = Column(Float)
    egfr = Column(Float)
    bmi = Column(Float)
    age = Column(Integer)
    nraas_drug_use = Column(Float)
    hypertension_history = Column(Float)
    prediction_percentage = Column(Integer)
    risk_level = Column(String(32))
    created_at = Column(DateTime, server_default=func.now())


# 2.2 激活码/邀请码表
class InviteCode(Base):
    __tablename__ = "invite_codes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(32), unique=True, nullable=False, index=True)
    is_used = Column(SmallInteger, default=0)  # 0: 未使用, 1: 已使用
    created_at = Column(DateTime, server_default=func.now())


# 2.3 授权白名单用户表
class AllowedUser(Base):
    __tablename__ = "allowed_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    _openid = Column(String(64), unique=True, nullable=False, index=True)
    doctor_name = Column(String(50), nullable=False)
    invite_code = Column(String(32))
    created_at = Column(DateTime, server_default=func.now())


# 全局 ONNX 模型句柄
sess = None


# 获取数据库 Session 依赖
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- 3. 授权验证依赖函数 ---
def verify_allowed_user(
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID"),
    db: Session = Depends(get_db)
):
    """
    拦截未在白名单中的未授权用户
    """
    if not x_wx_openid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="无法识别微信用户身份，请从微信小程序端发起调用"
        )
    
    user = db.query(AllowedUser).filter(AllowedUser._openid == x_wx_openid).first()
    if not user:
        # 特殊 HTTP 403 提示，供前端捕获并自动弹出激活框
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="UNAUTHORIZED_USER"
        )
    return user


# --- 4. Lifespan 启动与模型初始化 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global sess
    # 4.1 初始化自动建表（predictions, invite_codes, allowed_users）
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("MySQL Database and tables loaded/initialized successfully.")
    except Exception as e:
        logger.error(f"Database table creation failed: {e}")

    # 4.2 加载 ONNX 模型
    try:
        model_path = os.path.join(os.path.dirname(__file__), "onnx_diffusion.onnx")
        sess = ort.InferenceSession(model_path)
        logger.info("ONNX Diffusion Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        sess = None

    yield
    logger.info("🛑 Service shutting down...")


app = FastAPI(title="Diffusion Model Risk Predictor", lifespan=lifespan)


# --- Pydantic 数据请求模型 ---
class PredictionData(BaseModel):
    cfbg: float
    cDBP: float
    eGFR: float
    bmi: float
    nraas_drug_use: float
    hypertension_history: float
    age: float = Field(..., ge=0, le=100)


class ActivateRequest(BaseModel):
    doctor_name: str
    invite_code: str


# --- 5. 路由接口定义 ---

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": sess is not None}


# 解决腾讯云健康检查 /__tcb_probe__ 报 404 的问题
@app.get("/__tcb_probe__")
def tcb_probe():
    return {"status": "ok"}


# 5.1 检查用户授权状态接口
@app.get("/user/check_auth")
async def check_user_auth(
    db: Session = Depends(get_db),
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    if not x_wx_openid:
        return {"code": 0, "is_authorized": False, "msg": "未获取到 OpenID"}
    
    user = db.query(AllowedUser).filter(AllowedUser._openid == x_wx_openid).first()
    if user:
        return {
            "code": 0, 
            "is_authorized": True, 
            "doctor_name": user.doctor_name
        }
    return {"code": 0, "is_authorized": False}


# 5.2 激活码绑定/激活接口
@app.post("/user/activate")
async def activate_user(
    req: ActivateRequest,
    db: Session = Depends(get_db),
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    if not x_wx_openid:
        raise HTTPException(status_code=400, detail="未获取到微信身份 (OpenID)")

    # 1. 检查当前 OpenID 是否已存在于白名单
    existing_user = db.query(AllowedUser).filter(AllowedUser._openid == x_wx_openid).first()
    if existing_user:
        return {"code": 0, "msg": "您已绑定授权，无需重复激活"}

    # 2. 校验激活码是否有效且未被占用
    clean_code = req.invite_code.strip()
    code_obj = db.query(InviteCode).filter(
        InviteCode.code == clean_code,
        InviteCode.is_used == 0
    ).first()

    if not code_obj:
        raise HTTPException(status_code=400, detail="激活码无效或已被使用，请核对后重试")

    try:
        # 3. 标记激活码已被使用
        code_obj.is_used = 1
        
        # 4. 将该医生绑定进 AllowedUser 白名单
        new_user = AllowedUser(
            _openid=x_wx_openid,
            doctor_name=req.doctor_name.strip(),
            invite_code=clean_code
        )
        db.add(new_user)
        db.commit()

        logger.info(f"Doctor [{req.doctor_name.strip()}] activated successfully with code [{clean_code}].")
        return {"code": 0, "msg": "系统激活成功！欢迎使用"}
    except Exception as e:
        db.rollback()
        logger.error(f"Activation failure: {e}")
        raise HTTPException(status_code=500, detail=f"激活失败: {str(e)}")


# 5.3 核心预测 API（受 `verify_allowed_user` 保护）
@app.post("/predict")
async def predict_data(
    data: PredictionData,
    db: Session = Depends(get_db),
    user: AllowedUser = Depends(verify_allowed_user)  # 👈 仅限已激活医生访问
):
    if sess is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="ONNX model is not loaded on server."
        )

    try:
        # 1. ONNX 扩散模型推理
        cond_input = np.array([[
            data.cfbg, data.cDBP, data.eGFR, data.bmi, 
            data.nraas_drug_use, data.hypertension_history, data.age
        ]]).astype(np.float32)

        outputs = sess.run(None, {"cond": cond_input, "t": np.array([500], dtype=np.float32)})
        output_fake = softmax(outputs[0], axis=1)
        output_val = float(output_fake[0, 1])

        if output_val > 0.692:
            risk = "High Risk"
        elif output_val > 0.515:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        percentage_result = int(round(output_val * 100))

        # 2. 保存至数据库
        try:
            record = PredictionRecord(
                _openid=user._openid,  # 使用激活校验返回的医生 OpenID
                cfbg=data.cfbg,
                cdbp=data.cDBP,
                egfr=data.eGFR,
                bmi=data.bmi,
                age=int(data.age),
                nraas_drug_use=data.nraas_drug_use,
                hypertension_history=data.hypertension_history,
                prediction_percentage=percentage_result,
                risk_level=risk
            )
            db.add(record)
            db.commit()
            logger.info(f"Prediction result saved for OpenID: {user._openid}")
        except Exception as db_err:
            db.rollback()
            logger.error(f"Failed to save prediction record: {db_err}")

        return {
            "code": 0,
            "msg": "success",
            "prediction_percentage": percentage_result,
            "risk_level": risk
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")


# 5.4 获取以往评估历史记录（受 `verify_allowed_user` 保护）
@app.get("/history")
async def get_history(
    db: Session = Depends(get_db),
    user: AllowedUser = Depends(verify_allowed_user)  # 👈 仅限已激活医生访问
):
    try:
        records = db.query(PredictionRecord).filter(
            PredictionRecord._openid == user._openid
        ).order_by(PredictionRecord.created_at.desc()).limit(50).all()

        result = []
        for r in records:
            result.append({
                "id": r.id,
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "",
                "cfbg": r.cfbg,
                "cdbp": r.cdbp,
                "egfr": r.egfr,
                "bmi": r.bmi,
                "age": r.age,
                "hypertension_history": r.hypertension_history,
                "prediction_percentage": r.prediction_percentage,
                "risk_level": r.risk_level
            })

        return {"code": 0, "data": result}
    except Exception as e:
        logger.error(f"Query history failed: {e}")
        return {"code": 0, "data": []}


# 5.5 清空当前医生的评估历史记录（受 `verify_allowed_user` 保护）
@app.delete("/history/clear")
async def clear_history(
    db: Session = Depends(get_db),
    user: AllowedUser = Depends(verify_allowed_user)  # 👈 仅限已激活医生访问
):
    try:
        deleted_count = db.query(PredictionRecord).filter(
            PredictionRecord._openid == user._openid
        ).delete()
        
        db.commit()
        return {"code": 0, "msg": f"成功清空 {deleted_count} 条历史记录"}
    except Exception as e:
        db.rollback()
        logger.error(f"Clear history failed: {e}")
        raise HTTPException(status_code=500, detail=f"清空记录失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)