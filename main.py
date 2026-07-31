import os
import numpy as np
import onnxruntime as ort
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Header, status, Depends
from pydantic import BaseModel, Field
from scipy.special import softmax
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session

app = FastAPI(title="Diffusion Model Risk Predictor - WeChat CloudBase")

# --- 数据库连接配置 (微信云托管环境变量自动注入) ---
# 环境变量读取，若本地测试可 fallback 到默认值
MYSQL_USERNAME = os.getenv("MYSQL_USERNAME", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_ADDRESS = os.getenv("MYSQL_ADDRESS", "127.0.0.1:3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "custom_db")

# 格式化数据库链接
DATABASE_URL = f"mysql+pymysql://{MYSQL_USERNAME}:{MYSQL_PASSWORD}@{MYSQL_ADDRESS}/{MYSQL_DATABASE}?charset=utf8mb4"

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 定义 ORM 表模型
class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    openid = Column(String(64), index=True)
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

# 自动建表
Base.metadata.create_all(bind=engine)

# 获取数据库 Session 依赖
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- ONNX 模型初始化 ---
try:
    sess = ort.InferenceSession("onnx_diffusion.onnx")
    print("✅ ONNX Model Loaded Successfully.")
except Exception as e:
    print(f"❌ Model Load Error: {e}")
    sess = None

class PredictionData(BaseModel):
    cfbg: float
    cDBP: float
    eGFR: float
    bmi: float
    nraas_drug_use: float
    hypertension_history: float
    age: float = Field(..., ge=0, le=100)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "FastAPI with CloudBase MySQL running"}

# --- 1. 预测并直接存入云托管数据库 ---
@app.post("/predict")
async def predict_data(
    data: PredictionData,
    db: Session = Depends(get_db),
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    if sess is None:
        raise HTTPException(status_code=500, detail="ONNX model is not loaded.")

    try:
        # 模型推理
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

        # 将预测记录自动写入 MySQL 数据库
        record = PredictionRecord(
            openid=x_wx_openid or "anonymous",
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

        return {
            "code": 0,
            "msg": "success",
            "prediction_percentage": percentage_result,
            "risk_level": risk
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Inference/Database Error: {str(e)}")

# --- 2. 新增：从云托管数据库获取当前用户的以往历史数据 ---
@app.get("/history")
async def get_history(
    db: Session = Depends(get_db),
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    user_openid = x_wx_openid or "anonymous"
    records = db.query(PredictionRecord).filter(
        PredictionRecord.openid == user_openid
    ).order_by(PredictionRecord.created_at.desc()).limit(50).all()

    result = []
    for r in records:
        result.append({
            "id": r.id,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "",
            "cfbg": r.cfbg,
            "cDBP": r.cdbp,
            "eGFR": r.egfr,
            "bmi": r.bmi,
            "age": r.age,
            "hypertension_history": r.hypertension_history,
            "prediction_percentage": r.prediction_percentage,
            "risk_level": r.risk_level
        })

    return {"code": 0, "data": result}