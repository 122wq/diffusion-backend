import os
import numpy as np
import onnxruntime as ort
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, status, Depends
from pydantic import BaseModel, Field
from scipy.special import softmax
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# --- 1. 数据库连接配置 (环境变量防护) ---
MYSQL_USERNAME = os.getenv("MYSQL_USERNAME", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_ADDRESS = os.getenv("MYSQL_ADDRESS", "172.17.0.7:3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "custom_db")

DATABASE_URL = f"mysql+pymysql://{MYSQL_USERNAME}:{MYSQL_PASSWORD}@{MYSQL_ADDRESS}/{MYSQL_DATABASE}?charset=utf8mb4"

# 设置合理的超时时间，防止阻塞服务启动
engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True, 
    pool_recycle=3600,
    connect_args={"connect_timeout": 5}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

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

# 全局模型句柄
sess = None

# --- 2. 使用 Lifespan 优雅管理启动过程 (防止容器崩溃) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global sess
    # 2.1 安全初始化数据库
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ MySQL Database connected & tables created successfully.")
    except Exception as e:
        print(f"⚠️ Database initialization failed (Will retry on API call): {e}")

    # 2.2 安全加载 ONNX 模型
    try:
        model_path = os.path.join(os.path.dirname(__file__), "onnx_diffusion.onnx")
        sess = ort.InferenceSession(model_path)
        print("✅ ONNX Diffusion Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading ONNX model: {e}")
        sess = None

    yield  # 服务在此处正常监听端口并处理请求

    # 服务停止时的清理操作
    print("🛑 Service shutting down...")

app = FastAPI(title="Diffusion Model Risk Predictor - WeChat CloudBase", lifespan=lifespan)

# 获取数据库 Session 依赖
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PredictionData(BaseModel):
    cfbg: float
    cDBP: float
    eGFR: float
    bmi: float
    nraas_drug_use: float
    hypertension_history: float
    age: float = Field(..., ge=0, le=100)

# 探针接口：确保即使数据库或模型暂时不可用，探针也能通（防止 K8s 杀容器）
@app.get("/")
def health_check():
    return {
        "status": "ok", 
        "model_loaded": sess is not None,
        "message": "FastAPI with CloudBase MySQL running"
    }

# --- 3. 预测接口 ---
@app.post("/predict")
async def predict_data(
    data: PredictionData,
    db: Session = Depends(get_db),
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    if sess is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="ONNX model is not loaded on server. Check server logs."
        )

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

        # 写入数据库记录
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

# --- 4. 历史接口 ---
@app.get("/history")
async def get_history(
    db: Session = Depends(get_db),
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch history failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)