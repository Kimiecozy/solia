# backend/app/test.py
# ===============================================
# SolIA
# Scoring Solvabilité
# M2 SEP
# ===============================================

import sys
from pathlib import Path
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Ajouter le projet au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import MLConfig
from ml_pipeline.preprocessing.feature_engineering import load_and_prepare_data, RecommendationFeatureEngine
from ml_pipeline.train_model import train_recommendation_model 

# Logging basique
logging.basicConfig(
    level=logging.INFO,
    format="[{asctime}] {levelname:<8} {name}: {message}",
    style="{"
)
logger = logging.getLogger(__name__)

# ----------------------
# Instance FastAPI
# ----------------------
app = FastAPI(
    title="Olist Recommendation API Test",
    version="1.0",
    description="API de test pour Scoring Solvabilité",
)

# Middleware CORS pour le frontend Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de sécurité hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0"]
)

# ----------------------
# Routes
# ----------------------
@app.get("/")
async def root():
    return {"message": "✅ Backend test OK"}

@app.get("/train")
async def train():
    """
    Lance l'entraînement du modèle de scoring.
    """
    try:
        data_dir = Path(MLConfig.RAW_DATA_DIR)  # ou RAW_DATA_DIR si défini
        metrics = train_recommendation_model(data_dir)
        return {
            "status": "success",
            "r2": metrics['r2'],
            "mae": metrics['mae']
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/seller_features")
async def seller_features():
    """
    Retourne les features des vendeurs (calculées à partir de ton pipeline).
    """
    try:
        data_dir = Path(MLConfig.RAW_DATA_DIR)
        sellers, orders, items, payments, reviews, products = load_and_prepare_data(data_dir)
        engine = RecommendationFeatureEngine()
        df_vendeurs = engine.create_seller_features(sellers, orders, items, payments, reviews, products)
        # Pour simplifier, on renvoie juste les 5 premières lignes
        return df_vendeurs.head(5).to_dict(orient="records")
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ----------------------
# Exception globale
# ----------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"❌ Erreur non gérée: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": str(exc)}
    )

# ----------------------
# Lancement uvicorn
# ----------------------
def main():
    uvicorn.run(
        "backend.app.test:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main()
