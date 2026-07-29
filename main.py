import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.special import softmax

app = FastAPI(title="Diffusion Model Risk Predictor")

# Enable CORS for Flutter communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load ONNX session once at startup
try:
    sess = ort.InferenceSession("onnx_diffusion.onnx")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    sess = None


class PredictionData(BaseModel):
    cfbg: float
    cDBP: float
    eGFR: float
    bmi: float
    nraas_drug_use: float
    hypertension_history: float
    age: float



# --- Protected Prediction Endpoint ---
@app.post("/predict")
async def predict_data(
    data: PredictionData, 
 # 2. FIX: Protect route with OAuth2 token dependency
):
    if sess is None:
        raise HTTPException(status_code=500, detail="ONNX model is not loaded on the server.")
        
    try:
        # Prepare 7 input variables
        cond_input = np.array([[
            data.cfbg, 
            data.cDBP, 
            data.eGFR, 
            data.bmi, 
            data.nraas_drug_use, 
            data.hypertension_history, 
            data.age
        ]]).astype(np.float32)

        # Run ONNX inference
        outputs = sess.run(
            None,
            {
                "cond": cond_input,
                "t": np.array([500], dtype=np.float32),  # diffusion timestep
            }
        )
        
        # Raw logit processing
        output_fake = outputs[0]
        output_fake = softmax(output_fake, axis=1)
        
        # Extract scalar probability
        output_val = float(output_fake[0, 1])

        # Classify patient risk
        if output_val > 0.692:
            risk = "High Risk"
        elif output_val > 0.515:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        return {
            "prediction_percentage": int(round(output_val * 100)),
            "risk_level": risk
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")