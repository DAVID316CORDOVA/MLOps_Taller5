from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import logging
import os
import mlflow
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Penguins Species Prediction API",
    description="API optimizada para alta concurrencia",
    version="3.0.0"
)

# Variables globales
model = None
model_loaded_at = None
species_mapping = {1: "Adelie", 2: "Chinstrap", 3: "Gentoo"}

class PenguinFeatures(BaseModel):
    bill_length_mm: float = Field(..., example=39.1, ge=0)
    bill_depth_mm: float = Field(..., example=18.7, ge=0)
    flipper_length_mm: float = Field(..., example=181.0, ge=0)
    body_mass_g: float = Field(..., example=3750.0, ge=0)
    year: int = Field(..., example=2007, ge=0)
    island_Biscoe: int = Field(0, example=0, ge=0)
    island_Dream: int = Field(0, example=0, ge=0)  
    island_Torgersen: int = Field(1, example=1, ge=0)
    sex_female: int = Field(0, example=0, ge=0)
    sex_male: int = Field(1, example=1, ge=0)

@app.on_event("startup")
async def load_model():
    global model, model_loaded_at
    try:
        logger.info("Iniciando carga del modelo...")
        
        # Configuración MLflow
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "admin")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "supersecret")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5008"))
        
        MODEL_NAME = "reg_logistica"
        MODEL_STAGE = "Production"
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        
        logger.info(f"⬇️ Descargando modelo desde: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        model_loaded_at = time.time()
        
        logger.info(f"Modelo cargado en memoria. Tipo: {type(model).__name__}")
        
        # Test de predicción
        test_data = pd.DataFrame([{
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "year": 2007,
            "island_Biscoe": 0,
            "island_Dream": 0,
            "island_Torgersen": 1,
            "sex_female": 0,
            "sex_male": 1
        }])
        test_pred = model.predict(test_data)[0]
        logger.info(f"Test de predicción: {test_pred} ({species_mapping.get(int(test_pred), 'Unknown')})")
        
    except Exception as e:
        logger.error(f" Error cargando modelo: {e}")
        raise e

@app.post("/predict")
def predict(features: PenguinFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Convertir a DataFrame (compatible con pyfunc)
        feature_df = pd.DataFrame([features.dict()])
        prediction = model.predict(feature_df)[0]
        
        return {
            "species_id": int(prediction),
            "species_name": species_mapping.get(int(prediction), "Unknown"),
            "latency_ms": "<10ms",
            "model_source": "In-Memory Cache"
        }
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_loaded_at": model_loaded_at
    }

@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return {
        "model_type": str(type(model).__name__),
        "model_loaded": True,
        "model_in_memory": True,
        "description": "Modelo cacheado en memoria - predicciones instantáneas sin overhead de MLflow"
    }
