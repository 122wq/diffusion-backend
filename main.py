import os
import logging
import requests
import numpy as np
import onnxruntime as ort
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, status
from pydantic import BaseModel, Field
from scipy.special import softmax

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 全局 ONNX 模型句柄
sess = None

# --- 1. 读取微信云托管环境变量 ---
# 云托管在绑定数据库后会自动注入 MYSQL_ADDRESS 等配置
MYSQL_ADDRESS = os.getenv("MYSQL_ADDRESS", "127.0.0.1:3306")
TABLE_NAME = "predictions"

# 如果使用内网 HTTP REST 服务点，通常为 http://<MYSQL_ADDRESS_IP>/v1/rdb/rest/...
# 这里演示通过 requests 通用 HTTP 的处理逻辑
REST_BASE_URL = os.getenv("CLOUDBASE_REST_URL", f"http://{MYSQL_ADDRESS.split(':')[0]}:8080/v1/rdb/rest")

# --- 2. 使用 Lifespan 管理启动与模型加载 ---
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

app = FastAPI(title="Diffusion Model Risk Predictor - CloudBase REST", lifespan=lifespan)

# 请求体数据结构校验
class PredictionData(BaseModel):
    cfbg: float
    cDBP: float
    eGFR: float
    bmi: float
    nraas_drug_use: float
    hypertension_history: float
    age: float = Field(..., ge=0, le=100)

# --- 3. 数据库操作函数 (基于标准 requests) ---

def save_prediction_to_cloudbase(record_data: Dict[str, Any]) -> bool:
    """使用 requests 提交预测结果到云托管数据库 REST 接口"""
    url = f"{REST_BASE_URL}/{TABLE_NAME}"
    try:
        # 设置短超时，防止影响主流程响应
        resp = requests.post(url, json=record_data, timeout=3)
        if resp.status_code in (200, 201):
            logger.info("✅ 数据成功存入 CloudBase")
            return True
        else:
            logger.warning(f"⚠️ 数据库返回非 200 响应: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        logger.error(f"❌ 请求 CloudBase REST API 异常: {str(e)}")
        return False

def get_history_from_cloudbase(openid: str, limit: int = 50) -> List[Dict[str, Any]]:
    """使用 requests 从云托管数据库 REST 接口查询历史记录"""
    url = f"{REST_BASE_URL}/{TABLE_NAME}"
    params = {
        "limit": limit,
        "order": "created_at desc",
        "where": f"openid='{openid}'"
    }
    try:
        resp = requests.get(url, params=params, timeout=3)
        if resp.status_code == 200:
            res_json = resp.json()
            if isinstance(res_json, dict) and "data" in res_json:
                return res_json.get("data", [])
            elif isinstance(res_json, list):
                return res_json
        return []
    except Exception as e:
        logger.error(f"❌ 查询历史记录异常: {str(e)}")
        return []

# --- 4. API 路由接口 ---

@app.get("/")
def health_check():
    """健康检查探针"""
    return {
        "status": "ok", 
        "model_loaded": sess is not None
    }

@app.post("/predict")
async def predict_data(
    data: PredictionData,
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    """风险模型推理并将记录存入 CloudBase"""
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

        # 2. 组装记录
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
        
        # 3. 保存记录（捕获异常，确保即使数据库不可用也不阻断推理结果返回）
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