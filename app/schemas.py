"""Schemas Pydantic que definem o contrato de entrada e saída da API.

A validação acontece na borda da aplicação: dados fora dos limites
observados no treinamento são rejeitados com HTTP 422 antes de chegarem
ao modelo de machine learning.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Limites derivados do dataset de treinamento (ver scripts/train.py)
MIN_TEMPO_ASSINATURA: float = 0.0
MAX_TEMPO_ASSINATURA: float = 5.0
MIN_FREQUENCIA_PEDIDOS: float = 0.0
MAX_FREQUENCIA_PEDIDOS: float = 50.0
MIN_VALOR_MEDIO: float = 0.0
MAX_VALOR_MEDIO: float = 1000.0


class ChurnFeatures(BaseModel):
    """Features de entrada para a predição de churn de um cliente."""

    tempo_assinatura: float = Field(
        ...,
        ge=MIN_TEMPO_ASSINATURA,
        le=MAX_TEMPO_ASSINATURA,
        description="Tempo de assinatura do cliente em anos.",
        examples=[2.5],
    )
    frequencia_pedidos: float = Field(
        ...,
        ge=MIN_FREQUENCIA_PEDIDOS,
        le=MAX_FREQUENCIA_PEDIDOS,
        description="Quantidade média de pedidos por mês.",
        examples=[18.0],
    )
    valor_medio: float = Field(
        ...,
        ge=MIN_VALOR_MEDIO,
        le=MAX_VALOR_MEDIO,
        description="Valor médio gasto por pedido, em reais.",
        examples=[75.5],
    )


class PredictionResponse(BaseModel):
    """Resultado da predição de churn."""

    churn: int = Field(
        ...,
        description="1 se o cliente tem propensão a cancelar, 0 caso contrário.",
        examples=[1],
    )
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade estimada de cancelamento (0 a 1).",
        examples=[0.82],
    )
    risk_level: Literal["baixo", "medio", "alto"] = Field(
        ...,
        description="Faixa de risco derivada da probabilidade.",
        examples=["alto"],
    )
