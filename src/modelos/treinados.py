"""Os modelos já treinados: o script de treino salva, a app só carrega e usa."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from modelos.regressao_trechos import ResultadoRegressao

PASTA = Path(__file__).resolve().parents[2] / "modelos_treinados"
ARQUIVO = PASTA / "modelos.joblib"

CLASSIFICADOR = "MLP (scikit-learn)"


@dataclass(frozen=True)
class ModelosTreinados:
    treinado_em: datetime
    base: str
    acidentes: int
    anos: str
    regressao: ResultadoRegressao | None
    classificador: BaseEstimator
    metricas_classificador: pd.Series  # no conjunto de teste do treino
    linhas_classificador: int


def salvar(modelos: ModelosTreinados) -> None:
    PASTA.mkdir(exist_ok=True)
    joblib.dump(modelos, ARQUIVO)


def carregar() -> ModelosTreinados | None:
    if not ARQUIVO.exists():
        return None

    return joblib.load(ARQUIVO)
