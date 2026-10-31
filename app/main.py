"""Ponto de entrada da API de predição de churn.

Este módulo apenas configura a aplicação FastAPI e expõe endpoints de
infraestrutura (identificação e health check). A integração com o modelo
de machine learning é adicionada na etapa seguinte.
"""

from fastapi import FastAPI

API_VERSION: str = "0.1.0"

app = FastAPI(
    title="Churn Predictor API",
    description=(
        "API que estima a propensão de cancelamento (churn) de clientes "
        "a partir de features comportamentais, usando um modelo "
        "RandomForestClassifier treinado com scikit-learn."
    ),
    version=API_VERSION,
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
