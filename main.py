import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Header, status
from pydantic import BaseModel, Field
from scipy.special import softmax
from typing import Optional

app = FastAPI(title="Diffusion Model Risk Predictor - WeChat CloudBase")

# 1. 启动时加载 ONNX 模型文件
try:
    sess = ort.InferenceSession("onnx_diffusion.onnx")
    print("ONNX Diffusion Model loaded successfully.")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    sess = None


# 2. 定义输入数据格式
class PredictionData(BaseModel):
    cfbg: float
    cDBP: float
    eGFR: float
    bmi: float
    nraas_drug_use: float
    hypertension_history: float
    age: float = Field(..., ge=0, le=100)


# 3. 健康检查探针
@app.get("/")
def health_check():
    return {
        "status": "ok", 
        "model_loaded": sess is not None,
        "message": "Diffusion Model API is running"
    }


# 4. 预测接口 (适配微信云托管)
@app.post("/predict")
async def predict_data(
    data: PredictionData,
    # 微信云托管会自动注入 OpenID，代替原有的 OAuth2 Token
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    if sess is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="ONNX model is not loaded on the server."
        )

    try:
        # 准备 7 个临床特征输入
        cond_input = np.array([[
            data.cfbg,
            data.cDBP,
            data.eGFR,
            data.bmi,
            data.nraas_drug_use,
            data.hypertension_history,
            data.age
        ]]).astype(np.float32)

        # 运行 ONNX 扩散模型推理 (带 Timestep = 500)
        outputs = sess.run(
            None,
            {
                "cond": cond_input,
                "t": np.array([500], dtype=np.float32),  # diffusion timestep
            }
        )

        # 计算 Raw Logits 的 Softmax
        output_fake = outputs[0]
        output_fake = softmax(output_fake, axis=1)

        # 提取高风险类别 (索引 1) 的概率标量值 (0.0 ~ 1.0)
        output_val = float(output_fake[0, 1])

        # 临界值风险分类
        if output_val > 0.692:
            risk = "High Risk"
        elif output_val > 0.515:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        # 正确转化为 0 - 100 的整数百分比
        percentage_result = int(round(output_val * 100))

        return {
            "code": 0,
            "msg": "success",
            "user_openid": x_wx_openid,
            "prediction_percentage": percentage_result,
            "risk_level": risk
        }

    except Exception as e:
        print(f"Inference error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Inference failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)