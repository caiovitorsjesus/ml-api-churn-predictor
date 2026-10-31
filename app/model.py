"""Camada de inferência: carregamento do modelo e cálculo da predição.

Mantém a lógica de machine learning isolada da camada HTTP, de modo que
possa ser testada sem subir um servidor.
"""

import logging
import os
from typing import Final, Literal

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.schemas import ChurnFeatures, PredictionResponse

logger = logging.getLogger(__name__)

MODEL_PATH: Final[str] = os.getenv(
    "MODEL_PATH", os.path.join("app", "churn_model.joblib")
)

# Limiares de negócio para traduzir probabilidade em faixa de risco
THRESHOLD_BAIXO: Final[float] = 0.30
THRESHOLD_ALTO: Final[float] = 0.70
DECISION_THRESHOLD: Final[float] = 0.50


def load_model(model_path: str = MODEL_PATH) -> RandomForestClassifier:
    """Carrega o modelo treinado do disco.

    Args:
        model_path: Caminho do arquivo .joblib.

    Returns:
        O classificador pronto para inferência.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo não encontrado em '{model_path}'. "
            "Execute 'python scripts/train.py' para gerá-lo."
        )

    model = joblib.load(model_path)
    logger.info("Modelo carregado de %s", model_path)
    return model


def classify_risk(probability: float) -> Literal["baixo", "medio", "alto"]:
    """Converte a probabilidade de churn em uma faixa de risco de negócio."""
    if probability >= THRESHOLD_ALTO:
        return "alto"
    if probability >= THRESHOLD_BAIXO:
        return "medio"
    return "baixo"


def predict(model: RandomForestClassifier, features: ChurnFeatures) -> PredictionResponse:
    """Executa a inferência para um cliente.

    Args:
        model: Classificador já carregado.
        features: Features validadas do cliente.

    Returns:
        A predição com probabilidade e faixa de risco.
    """
    payload = features.model_dump()

    # Usa a ordem de colunas que o modelo aprendeu no treinamento,
    # evitando o warning de feature names do scikit-learn.
    columns = list(getattr(model, "feature_names_in_", payload.keys()))
    row = pd.DataFrame([[payload[column] for column in columns]], columns=columns)

    probability = float(model.predict_proba(row)[0][1])
    churn = int(probability >= DECISION_THRESHOLD)

    return PredictionResponse(
        churn=churn,
        churn_probability=round(probability, 4),
        risk_level=classify_risk(probability),
    )
