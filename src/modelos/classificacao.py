from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from avaliacao import dividir_treino_teste, metricas
from colunas import ALVO, CATEGORICAS
from modelos.treino import treinar

# acima disso, o treino na tela demora demais, mas isso mantém a proporção de fatais
MAXIMO_LINHAS = 100_000

MINIMO_FATAIS = 10


@dataclass(frozen=True)
class ResultadoClassificacao:
    metricas: pd.DataFrame
    linhas_usadas: int
    odds_pista_simples: float | None


def _odds_pista_simples(pipeline) -> float | None:
    # razão de chances de um acidente fatal: pista simples contra pista dupla, com as outras variáveis fixas
    onehot = pipeline.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    nomes = list(onehot.get_feature_names_out(list(CATEGORICAS)))
    coeficientes = pipeline.named_steps["modelo"].coef_[0]

    if "tipo_pista_Simples" not in nomes or "tipo_pista_Dupla" not in nomes:
        return None

    # as colunas categóricas vêm primeiro na saída do ColumnTransformer
    diferenca = coeficientes[nomes.index("tipo_pista_Simples")] - coeficientes[nomes.index("tipo_pista_Dupla")]
    return float(np.exp(diferenca))


def avaliar_classificadores(df: pd.DataFrame, nomes: list[str], maximo: int = MAXIMO_LINHAS) -> ResultadoClassificacao | None:
    fatais = int(df[ALVO].sum())
    if fatais < MINIMO_FATAIS or fatais == len(df):
        return None

    if len(df) > maximo:
        df, _ = train_test_split(df, train_size=maximo, stratify=df[ALVO], random_state=2903)

    divisao = dividir_treino_teste(df)
    linhas = {}
    odds = None

    for nome in nomes:
        pipeline = treinar(nome, divisao.treino)
        probabilidades = pipeline.predict_proba(divisao.teste)[:, 1]
        linhas[nome] = metricas(divisao.teste[ALVO], probabilidades)

        if nome == "Regressão logística":
            odds = _odds_pista_simples(pipeline)

    return ResultadoClassificacao(
        metricas=pd.DataFrame.from_dict(linhas, orient="index"),
        linhas_usadas=len(df),
        odds_pista_simples=odds,
    )
