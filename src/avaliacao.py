from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from colunas import ALVO, VAZAMENTO
from modelos.treino import treinar

import numpy as np
import pandas as pd


FATIAS = 5


# TODO: adicionar aqui enquanto cada pessoa for adicionando as partes dela de limpeza de colunas
def montar_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(VAZAMENTO), errors="ignore")


def validacao_cruzada(nome: str, df: pd.DataFrame, fatias: int = FATIAS) -> pd.DataFrame:
    df = montar_features(df)
    divisor = StratifiedKFold(n_splits=fatias, shuffle=True, random_state=2903)

    linhas = []
    for indices_treino, indices_teste in divisor.split(df, df[ALVO]):
        treino, teste = df.iloc[indices_treino], df.iloc[indices_teste]
        modelo = treinar(nome, treino)
        linhas.append(metricas(teste[ALVO], modelo.predict_proba(teste)[:, 1]))

    return pd.DataFrame(linhas)  # uma linha por fatia


def metricas(y_verdadeiro, probabilidades, limiar: float = 0.5) -> dict[str, float]:
    y_verdadeiro = np.asarray(y_verdadeiro, dtype=bool)
    probabilidades = np.asarray(probabilidades, dtype=float)
    
    y_previsto = probabilidades >= limiar

    if y_verdadeiro.any():
        precisao_media = average_precision_score(y_verdadeiro, probabilidades)
    else:
        precisao_media = float("nan")

    return {
        "recall": recall_score(y_verdadeiro, y_previsto, zero_division=0),
        "precisao": precision_score(y_verdadeiro, y_previsto, zero_division=0),
        "f1": f1_score(y_verdadeiro, y_previsto, zero_division=0),
        "precisao_media": precisao_media,
        "acuracia": accuracy_score(y_verdadeiro, y_previsto),
    } # type: ignore
