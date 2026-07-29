from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="微信云托管心血管风险评估服务")

# 1. 定义请求数据格式（对应小程序的入参）
class RiskPredictRequest(BaseModel):
    cfbg: float = Field(..., description="Clinical Systolic BP / Glucose")
    cDBP: float = Field(..., description="Clinical Diastolic BP")
    eGFR: float = Field(..., description="Estimated GFR")
    bmi: float = Field(..., description="Body Mass Index")
    nraas_drug_use: float = Field(..., description="0.0 或 1.0")
    hypertension_history: float = Field(..., description="0.0 或 1.0")
    age: float = Field(..., ge=0, le=100, description="年龄 0-100")

# 2. 健康检查接口（云托管容器存活探针）
@app.get("/")
def health_check():
    return {"status": "ok", "message": "FastAPI on WeChat CloudBase is running!"}

# 3. 核心风险预测 API  
@app.post("/predict")
def predict_risk(
    payload: RiskPredictRequest,
    # 微信云托管会自动在请求头中注入用户的 OpenID，无需额外传 token 鉴权
    x_wx_openid: Optional[str] = Header(None, alias="X-WX-OPENID")
):
    try:
        # TODO: 这里替换为你训练好的机器学习模型预测逻辑 (如 model.predict)
        # 示例：简单逻辑计算模拟
        score = (payload.age * 0.3) + (payload.bmi * 0.4) + (payload.cDBP * 0.2)
        percentage = min(int(score), 99)

        if percentage > 70:
            risk_level = "High Risk"
        elif percentage > 40:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        return {
            "code": 0,
            "msg": "success",
            "user_openid": x_wx_openid, # 自动识别到的微信用户ID
            "prediction_percentage": percentage,
            "risk_level": risk_level
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)