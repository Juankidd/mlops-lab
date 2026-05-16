"""
CD Stage: ML Model Serving — FastAPI
ML Prediction Service del diagrama MLOps
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
import logging

app = FastAPI(title="Housing Price Prediction API", version="1.0")
logger = logging.getLogger(__name__)

# Carga el modelo al iniciar (del Model Registry)
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model("models:/housing-price-model/Production")
        logger.info("✅ Modelo Production cargado")
    except:
        # Fallback: cargar desde archivo local
        import joblib
        model = joblib.load("models/model.pkl")
        logger.warning("⚠️ Modelo cargado desde archivo local")

# Schema de entrada
class HouseFeatures(BaseModel):
    MedInc: float       # Ingreso medio del bloque
    HouseAge: float     # Edad media de casas
    AveRooms: float     # Habitaciones promedio
    AveBedrms: float    # Dormitorios promedio
    Population: float   # Población del bloque
    AveOccup: float     # Ocupación media
    Latitude: float
    Longitude: float

@app.post("/predict")
async def predict(features: HouseFeatures):
    """Predice el precio de una casa"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        df = pd.DataFrame([features.dict()])
        
        # Aplicar mismas features que en training
        df['rooms_per_person'] = df['AveRooms'] / (df['Population'] / df['AveOccup'] + 1)
        df['block_density'] = df['Population'] / (df['AveRooms'] + 1)
        df['income_squared'] = df['MedInc'] ** 2
        df['dist_to_center'] = np.sqrt(
            (df['Latitude'] - 37.77) ** 2 + (df['Longitude'] + 122.41) ** 2
        )
        df['is_bay_area'] = df['Latitude'].between(37, 38).astype(int)
        
        prediction = model.predict(df)[0]
        
        return {
            "predicted_price_100k": round(prediction, 4),
            "predicted_price_usd": round(prediction * 100000, 2),
            "timestamp": datetime.now().isoformat(),
            "model_version": "Production"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/docs")  # FastAPI genera Swagger automáticamente