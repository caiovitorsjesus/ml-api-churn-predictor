#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def generate_synthetic_data(n_samples=1000, random_state=42):
    """
    Gera dados sintéticos de exemplo para previsão de churn.
    
    Features:
    - tempo_assinatura (0-5 anos, maior reduz churn)
    - frequencia_pedidos (0-50, maior reduz churn)
    - valor_medio (0-1000, maior reduz churn)
    
    Target: 0 (não cancela), 1 (cancela)
    Lógica: Cliente cancela se (tempo*0.5 + freq*0.3) < threshold
    """
    np.random.seed(random_state)
    
    tempo_assinatura = np.random.uniform(0, 5, n_samples)
    frequencia_pedidos = np.random.uniform(0, 50, n_samples)
    valor_medio = np.random.uniform(0, 1000, n_samples)
    
    # Lógica de churn sintética
    score_risco = tempo_assinatura * 0.5 + frequencia_pedidos * 0.3
    threshold = 2.0  # Clientes com score de risco menor que 2 são menos propensos a cancelar
    churn = (score_risco < threshold).astype(int)
    
    df = pd.DataFrame({
        'tempo_assinatura': tempo_assinatura,
        'frequencia_pedidos': frequencia_pedidos,
        'valor_medio': valor_medio,
        'churn': churn
    })
    
    return df

def train_and_save_model(model_path='app/churn_model.joblib', n_samples=1000, random_state=42):
    """
    Gera dados, treina o modelo e salva o modelo treinado.
    """
    print("Gerando dados sintéticos...")
    df = generate_synthetic_data(n_samples=n_samples, random_state=random_state)
    
    # Features e target
    X = df[['tempo_assinatura', 'frequencia_pedidos', 'valor_medio']]
    y = df['churn']
    
    print("Realizando divisão dos dados...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    print("Treinando o modelo RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=random_state,
        n_jobs=-1  # Usa todos os cores disponíveis
    )
    model.fit(X_train, y_train)
    
    # Avaliação no conjunto de teste (para validação)
    accuracy = model.score(X_test, y_test)
    print(f"\nAvaliação do modelo no conjunto de teste:")
    print(f"Acurácia: {accuracy:.4f}")
    
    # Certifica que o diretório 'app' existe
    os.makedirs('app', exist_ok=True)
    
    # Salva o modelo treinado
    print(f"\nSalvando o modelo em {model_path}...")
    joblib.dump(model, model_path)
    print(f"Modelo salvo com sucesso!")
    
    # Mostra estatísticas básicas
    print("\nEstatísticas do dataset:")
    print(df.describe().to_string())

if __name__ == "__main__":
    train_and_save_model()