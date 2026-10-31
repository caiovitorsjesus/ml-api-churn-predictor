#!/usr/bin/env python3
"""
Script de treinamento e persistência de modelo de ML para previsão de churn.

Este script gera dados sintéticos, treina um classificador RandomForest 
e salva o modelo treinado para uso posterior em uma API FastAPI.
"""

import logging
import os
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constantes para reprodutibilidade e configuração
RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
N_ESTIMATORS: Final[int] = 100
MIN_SAMPLES_LEAF: Final[int] = 1
MAX_FEATURES: Final[str] = 'sqrt'
DEFAULT_MODEL_PATH: Final[str] = os.path.join('app', 'churn_model.joblib')

def generate_synthetic_data(n_samples: int = 1000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Gera dados sintéticos de exemplo para previsão de churn.
    
    Args:
        n_samples: Número de amostras a serem geradas.
        random_state: Semente para reprodutibilidade.
        
    Returns:
        pd.DataFrame: Dataset contendo as features e o alvo (churn).
    """
    rng = np.random.default_rng(random_state)

    tempo_assinatura = rng.uniform(0, 5, n_samples)
    frequencia_pedidos = rng.uniform(0, 50, n_samples)
    valor_medio = rng.uniform(0, 1000, n_samples)
    
    # Lógica de churn sintética: score de risco baseado em comportamento
    score_risco = tempo_assinatura * 0.5 + frequencia_pedidos * 0.3
    threshold = 2.0
    churn = (score_risco < threshold).astype(int)
    
    return pd.DataFrame({
        'tempo_assinatura': tempo_assinatura,
        'frequencia_pedidos': frequencia_pedidos,
        'valor_medio': valor_medio,
        'churn': churn
    })

def train_model(x_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    """Treina o modelo RandomForest."""
    logger.info("Treinando o modelo RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(x_train, y_train)
    return model

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """Persiste o modelo em disco."""
    try:
        directory = os.path.dirname(model_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        logger.info("Salvando o modelo em %s...", model_path)
        joblib.dump(model, model_path)
        logger.info("Modelo salvo com sucesso!")
    except (OSError, IOError) as e:
        logging.exception("Erro ao salvar o modelo no caminho %s: %s", model_path, e)
        raise

def train_and_save_model(
    model_path: str = DEFAULT_MODEL_PATH, 
    n_samples: int = 1000, 
    random_state: int = RANDOM_STATE
) -> None:
    """
    Fluxo principal: Gera dados, treina o modelo e persiste em disco.
    """
    logger.info("Iniciando processo de treinamento...")
    
    df = generate_synthetic_data(n_samples=n_samples, random_state=random_state)
    
    features = ['tempo_assinatura', 'frequencia_pedidos', 'valor_medio']
    x_data = df[features]
    y_data = df['churn']
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=TEST_SIZE, random_state=random_state
    )
    
    model = train_model(x_train, y_train, random_state=random_state)
    
    accuracy = model.score(x_test, y_test)
    logger.info("Acurácia no conjunto de teste: %.4f", accuracy)
    
    save_model(model, model_path)

if __name__ == "__main__":
    train_and_save_model()
