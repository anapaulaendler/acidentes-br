from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from colunas import ALVO, VAZAMENTO
from modelos.treino import treinar

import numpy as np
import pandas as pd


FATIAS = 5
TAMANHO_TESTE = 0.2
# as métricas de treino usam uma amostra: prever os 545 mil acidentes do treino com o KNN levaria horas
AMOSTRA_TREINO = 50_000


def dividir_treino_teste(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa 20% dos acidentes para o teste, com a mesma proporção de fatais. O teste não entra na validação cruzada."""
    return train_test_split(df, test_size=TAMANHO_TESTE, stratify=df[ALVO], random_state=2903)


def amostra_de_treino(treino: pd.DataFrame) -> pd.DataFrame:
    if len(treino) <= AMOSTRA_TREINO:
        return treino

    amostra, _ = train_test_split(treino, train_size=AMOSTRA_TREINO, stratify=treino[ALVO], random_state=2903)
    return amostra


def montar_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(VAZAMENTO), errors="ignore")


def validacao_cruzada(nome: str, df: pd.DataFrame, fatias: int = FATIAS, ajustes: dict | None = None, **params) -> pd.DataFrame:
    df = montar_features(df)
    divisor = StratifiedKFold(n_splits=fatias, shuffle=True, random_state=2903)

    linhas = []
    for indices_treino, indices_teste in divisor.split(df, df[ALVO]):
        treino, teste = df.iloc[indices_treino], df.iloc[indices_teste]
        modelo = treinar(nome, treino, ajustes, **params)
        linhas.append(metricas(teste[ALVO], modelo.predict_proba(teste)[:, 1]))

    return pd.DataFrame(linhas)  # uma linha por fatia


def metricas(y_verdadeiro, probabilidades, limiar: float = 0.5) -> dict[str, float]:
    y_verdadeiro = np.asarray(y_verdadeiro, dtype=bool)
    probabilidades = np.asarray(probabilidades, dtype=float)
    
    y_previsto = probabilidades >= limiar

    # sem as duas classes, a PR-AUC e o ROC-AUC não existem
    if y_verdadeiro.any() and not y_verdadeiro.all():
        precisao_media = average_precision_score(y_verdadeiro, probabilidades)
        roc_auc = roc_auc_score(y_verdadeiro, probabilidades)
    else:
        precisao_media = roc_auc = float("nan")

    return {
        "recall": recall_score(y_verdadeiro, y_previsto, zero_division=0),
        "precisao": precision_score(y_verdadeiro, y_previsto, zero_division=0),
        "f1": f1_score(y_verdadeiro, y_previsto, zero_division=0),
        "precisao_media": precisao_media,
        "roc_auc": roc_auc,
        "acuracia": accuracy_score(y_verdadeiro, y_previsto),
    } # type: ignore
