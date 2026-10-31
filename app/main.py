"""Ponto de entrada da API de predição de churn.

O modelo é carregado uma única vez no startup da aplicação (lifespan) e
reutilizado por todas as requisições.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request

from app.model import load_model, predict
from app.schemas import ChurnFeatures, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_VERSION: str = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo no startup e libera os recursos no shutdown."""
    app.state.model = load_model()
    yield
    app.state.model = None


app = FastAPI(
    title="Churn Predictor API",
    description=(
        "API que estima a propensão de cancelamento (churn) de clientes "
        "a partir de features comportamentais, usando um modelo "
        "RandomForestClassifier treinado com scikit-learn."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


@app.get("/", tags=["Infra"])
def read_root() -> dict[str, str]:
    """Identifica o serviço e aponta para a documentação interativa."""
    return {
        "service": "churn-predictor-api",
        "version": API_VERSION,
        "docs": "/docs",
    }


@app.get("/health", tags=["Infra"])
def health_check() -> dict[str, str]:
    """Health check usado por plataformas de deploy para validar o container."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["Predição"])
def predict_churn(features: ChurnFeatures, request: Request) -> PredictionResponse:
    """Estima a propensão de cancelamento de um cliente."""
    return predict(request.app.state.model, features)
