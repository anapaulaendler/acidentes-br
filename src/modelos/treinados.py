"""Os modelos já treinados: o script de treino salva, a app só carrega e usa."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

PASTA = Path(__file__).resolve().parents[2] / "modelos_treinados"
ARQUIVO = PASTA / "modelos.joblib"

CLASSIFICADORES = ["Árvore de decisão", "KNN", "MLP (scikit-learn)", "Rede com embeddings (PyTorch)", "Rede com one-hot (PyTorch)"]


@dataclass(frozen=True)
class Classificador:
    modelo: BaseEstimator
    metricas_fatias: pd.DataFrame  # uma linha por fatia
    # os hiperparâmetros escolhidos na tela; None = os padrões do catálogo (treino pelo script)
    parametros: dict | None = None


@dataclass(frozen=True)
class ModelosTreinados:
    treinado_em: datetime
    base: str
    acidentes: int
    anos: str
    classificadores: dict[str, Classificador]


def salvar(modelos: ModelosTreinados) -> None:
    PASTA.mkdir(exist_ok=True)
    joblib.dump(modelos, ARQUIVO, compress=3)


def carregar() -> ModelosTreinados | None:
    if not ARQUIVO.exists():
        return None

    return joblib.load(ARQUIVO)
