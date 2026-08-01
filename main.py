import os
import logging
import numpy as np
import onnxruntime as ort
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, status
from pydantic import BaseModel, Field
from scipy.special import softmax

# 引入 CloudBase Client SDK
try:
    from cloudbase_client import cloudbase
except ImportError:
    cloudbase = None

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 全局 ONNX 模型句柄
sess = None

# --- 1. 使用 Lifespan 管理启动与模型加载 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global sess
    # 安全加载 ONNX 扩散模型
    try:
        model_path = os.path.join(os.path.dirname(__file__), "onnx_diffusion.onnx")
        sess = ort.InferenceSession(model_path)
        logger.info("✅ ONNX Diffusion Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Error loading ONNX model: {e}")
        sess = None

    yield
    logger.info("🛑 Service shutting down...")

app = FastAPI(title="Diffusion Model Risk Predictor - CloudBase RDB", lifespan=lifespan)

# 请求体数据结构校验
class PredictionData(BaseModel):
    cfbg: float
    cDBP: float
    eGFR: float
    bmi: float
    nraas_drug_use: float
    hypertension_history: float
    age: float = Field(..., ge=0, le=100)

# --- 2. 数据库操作函数 (基于 CloudBase SDK) ---
TABLE_NAME = "predictions"

def save_prediction_to_cloudbase(record_data: Dict[str, Any]) -> bool:
    """通过 CloudBase Rest API 保存数据记录"""
    if cloudbase is None:
        logger.warning("⚠️ cloudbase_client SDK 未安装，跳过数据库写入")
        return False
    
    try:
        endpoint = f"/v1/rdb/rest/{TABLE_NAME}"
        # 发送 POST 请求插入记录
        response = cloudbase.request("POST", endpoint, data=record_data)
        logger.info(f"✅ 数据成功写入 CloudBase: {response}")
        return True
    except Exception as e:
        logger.error(f"❌ 写入 CloudBase 数据库失败: {str(e)}")
        return False

def get_history_from_cloudbase(openid: str, limit: int = 50) -> List[Dict[str, Any]]:
    """通过 CloudBase Rest API 获取用户的历史记录"""
    if cloudbase is None:
        logger.warning("⚠️ cloudbase_client SDK 未安装，无法读取历史")
        return []

    try:
        endpoint = f"/v1/rdb/rest/{TABLE_NAME}"
        params = {
            "limit": limit,
            "order": "created_at desc",
            "where": f"openid='{openid}'"  # 按 OpenID 筛选
        }
        response = cloudbase.request("GET", endpoint, params=params)
        
        if response and isinstance(response, dict) and "data" in response:
            return response.get("data", [])
        elif isinstance(response, list):
            return response
        return []
    except Exception as e:
        logger.error(f"❌ 从 CloudBase 读取历史失败: {str(e)}")
        return []

# --- 3. API 路由接口 ---

@app.get("/")
def health_check():
    """健康检查探针"""
    return {
        "status": "ok", 
        "model_loaded": sess is not None,
        "cloudbase_sdk_ready": cloudbase is not None
    }

@app.post("/predict")
async def predict_data(
    data: PredictionData,
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    """风险模型推理并异步存入 CloudBase"""
    if sess is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="ONNX model is not loaded on server."
        )

    try:
        # 1. 运算 ONNX 扩散模型
        cond_input = np.array([[
            data.cfbg, data.cDBP, data.eGFR, data.bmi, 
            data.nraas_drug_use, data.hypertension_history, data.age
        ]]).astype(np.float32)

        outputs = sess.run(None, {"cond": cond_input, "t": np.array([500], dtype=np.float32)})
        output_fake = softmax(outputs[0], axis=1)
        output_val = float(output_fake[0, 1])

        # 判断风险等级
        if output_val > 0.692:
            risk = "High Risk"
        elif output_val > 0.515:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        percentage_result = int(round(output_val * 100))

        # 2. 组装记录并存入 CloudBase 数据库
        record = {
            "openid": x_wx_openid or "anonymous",
            "cfbg": data.cfbg,
            "cdbp": data.cDBP,
            "egfr": data.eGFR,
            "bmi": data.bmi,
            "age": int(data.age),
            "nraas_drug_use": data.nraas_drug_use,
            "hypertension_history": data.hypertension_history,
            "prediction_percentage": percentage_result,
            "risk_level": risk
        }
        
        # 保存到数据库（如果失败仅记录日志，不阻断前端返回结果）
        save_prediction_to_cloudbase(record)

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
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    """获取以往评估数据"""
    user_openid = x_wx_openid or "anonymous"
    records = get_history_from_cloudbase(openid=user_openid)

    return {"code": 0, "data": records}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)