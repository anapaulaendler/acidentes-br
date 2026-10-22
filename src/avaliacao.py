from dataclasses import dataclass
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from colunas import ALVO, VAZAMENTO
from modelos.treino import treinar

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Divisao:
    treino: pd.DataFrame
    teste: pd.DataFrame


# TODO: adicionar aqui enquanto cada pessoa for adicionando as partes dela de limpeza de colunas
def montar_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(VAZAMENTO), errors="ignore")


def dividir_treino_teste(df: pd.DataFrame) -> Divisao:
    df = montar_features(df)

    treino, teste = train_test_split(df, test_size=0.2, random_state=2903, stratify=df[ALVO])

    if not treino[ALVO].any():
        raise ValueError("A base de treino está sem acidentes fatais.")

    return Divisao(treino=treino, teste=teste)


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


def comparar_modelos(nomes: list[str], divisao: Divisao, limiar: float = 0.5) -> pd.DataFrame:
    linhas = {}

    for nome in nomes:
        pipeline = treinar(nome, divisao.treino)
        probabilidades = pipeline.predict_proba(divisao.teste)[:, 1]
        
        linhas[nome] = metricas(divisao.teste[ALVO], probabilidades, limiar)

    return pd.DataFrame.from_dict(linhas, orient="index")
