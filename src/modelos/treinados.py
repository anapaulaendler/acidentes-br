"""Os modelos já treinados: o script de treino salva, o notebook só carrega e usa."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

PASTA = Path(__file__).resolve().parents[2] / "modelos_treinados"
ARQUIVO = PASTA / "modelos.joblib"

CLASSIFICADORES = [
    "Regressão logística", "Árvore de decisão", "KNN",  # clássicos
    "Random Forest", "Gradient Boosting",  # ensembles
    "MLP (scikit-learn)", "Rede com embeddings (PyTorch)", "Rede com one-hot (PyTorch)",  # redes neurais
]


@dataclass(frozen=True)
class Classificador:
    modelo: BaseEstimator
    metricas_fatias: pd.DataFrame  # a validação cruzada no treino: uma linha por fatia
    metricas_treino: dict[str, float]  # o modelo final numa amostra do treino
    metricas_teste: dict[str, float]  # o modelo final no teste, que ele nunca viu
    probabilidades_teste: np.ndarray  # a probabilidade de fatal de cada acidente do teste
    # os ajustes da busca (nomes do scikit-learn, como modelo__max_depth); {} ou None = os padrões do catálogo
    parametros: dict | None = None


@dataclass(frozen=True)
class ModelosTreinados:
    treinado_em: datetime
    base: str
    acidentes: int
    anos: str
    acidentes_treino: int
    alvo_teste: np.ndarray  # fatal ou não, na mesma ordem das probabilidades_teste
    classificadores: dict[str, Classificador]


def salvar(modelos: ModelosTreinados) -> None:
    PASTA.mkdir(exist_ok=True)
    joblib.dump(modelos, ARQUIVO, compress=3)


def carregar() -> ModelosTreinados | None:
    if not ARQUIVO.exists():
        return None

    return joblib.load(ARQUIVO)
