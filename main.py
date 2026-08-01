import os
import logging
import numpy as np
import onnxruntime as ort
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, status, Depends
from pydantic import BaseModel, Field
from scipy.special import softmax
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 1. 读取云托管自动注入的环境变量 ---

DB_HOST = os.getenv("DB_HOST", "sh-cynosdbmysql-grp-qj85u8vm.sql.tencentcdb.com")
DB_PORT = os.getenv("DB_PORT", "25802")
DB_USER = os.getenv("DB_USER", "jack")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")  # 必须能读取到正确密码
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

# 定义 ORM 表模型
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

# 全局模型句柄
sess = None

# 获取数据库 Session 依赖
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 2. 使用 Lifespan 管理启动与数据库建表 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global sess
    # 2.1 初始化建表
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ MySQL 数据库连接成功，数据表初始化完成。")
    except Exception as e:
        logger.error(f"⚠️ 数据库建表失败 (将在接口调用时重试): {e}")

    # 2.2 加载 ONNX 模型
    try:
        model_path = os.path.join(os.path.dirname(__file__), "onnx_diffusion.onnx")
        sess = ort.InferenceSession(model_path)
        logger.info("✅ ONNX Diffusion Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Error loading ONNX model: {e}")
        sess = None

    yield
    logger.info("🛑 Service shutting down...")

app = FastAPI(title="Diffusion Model Risk Predictor", lifespan=lifespan)

class PredictionData(BaseModel):
    cfbg: float
    cDBP: float
    eGFR: float
    bmi: float
    nraas_drug_use: float
    hypertension_history: float
    age: float = Field(..., ge=0, le=100)

# --- 3. 路由接口 ---

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": sess is not None}

# 解决腾讯云健康检查 /__tcb_probe__ 报 404 的问题
@app.get("/__tcb_probe__")
def tcb_probe():
    return {"status": "ok"}

@app.post("/predict")
async def predict_data(
    data: PredictionData,
    db: Session = Depends(get_db),
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    if sess is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="ONNX model is not loaded on server."
        )

    try:
        # 1. ONNX 模型推理
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

        # 2. 保存到 MySQL 数据库
        try:
            record = PredictionRecord(
                _openid=x_wx_openid or "anonymous",
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
            db.commit() # 🎯 执行真正的 SQL 写入
            logger.info("✅ 成功写入记录到 MySQL")
        except Exception as db_err:
            db.rollback()
            logger.error(f"⚠️ 写入数据库失败: {db_err}")

        return {
            "code": 0,
            "msg": "success",
            "prediction_percentage": percentage_result,
            "risk_level": risk
        }

    except Exception as e:
        logger.error(f"推理处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

@app.get("/history")
async def get_history(
    db: Session = Depends(get_db),
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    try:
        user_openid = x_wx_openid or "anonymous"
        records = db.query(PredictionRecord).filter(
            PredictionRecord._openid == user_openid
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
        logger.error(f"查询历史失败: {e}")
        return {"code": 0, "data": []}

# --- 3. 清空当前用户的历史记录 ---
@app.delete("/history/clear")
async def clear_history(
    db: Session = Depends(get_db),
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    user_openid = x_wx_openid or "anonymous"
    try:
        # 删除当前 OpenID 的所有预测记录
        deleted_count = db.query(PredictionRecord).filter(
            PredictionRecord._openid == user_openid
        ).delete()
        
        db.commit()
        return {"code": 0, "msg": f"成功清空 {deleted_count} 条历史记录"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"清空记录失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)