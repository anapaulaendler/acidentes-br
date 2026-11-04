import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from colunas import CATEGORICAS, NUMERICAS


DESCONHECIDO = "<desconhecido>"


def _para_texto(x):
    return x.astype(str)


def preprocessador() -> ColumnTransformer:
    categoricas = Pipeline([
        ("texto", FunctionTransformer(_para_texto)), 
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")), # inicial, depois avaliamos se é o melhor jeito
        ("padronizar", StandardScaler())
    ])

    return ColumnTransformer(
        [("cat", categoricas, list(CATEGORICAS)), ("num", numericas, NUMERICAS)], remainder="drop"
    )


# aqui basicamente a gente substitui o OneHot, que faz cada categoria virar 0 ou 1, por tipo um enum (aí UF ao invés de ser vários
# 0 ou 1 vira 1, 2, 3...). isso serve para o arquivo rede_neural, que precisa disso para a vetorização de posições em rede_neural.py. 
# a justificativa principal do porquê usar embedding ao invés do one-hot é que, no one-hot, temos a posição como algo parecido com: 
# "amanhecer" [0, 1, 0, 0, 0] e "anoitecer" [0, 0, 1, 0, 0], enquanto no embedding AS POSIÇÕES DO VETOR SÃO APRENDIDAS DURANTE O TREINO,
# O QUE AJUDA O MODELO A ENCONTRAR SIMILIARIDADES ENTRE ITENS CATEGÓRICOS. 
class CodificadorCategorias(BaseEstimator, TransformerMixin):
    def fit(self, df: pd.DataFrame, y=None) -> CodificadorCategorias:
        self.vocabularios_ = {}

        for coluna in CATEGORICAS:
            valores = sorted(df[coluna].dropna().astype(str).unique())

            # basicamente valores que não apareceram no treino
            vocabulario = {DESCONHECIDO: 0}

            for indice, valor in enumerate(valores, start=1):
                vocabulario[valor] = indice

            self.vocabularios_[coluna] = vocabulario

        self.cardinalidades_ = []

        for coluna in CATEGORICAS:
            self.cardinalidades_.append(len(self.vocabularios_[coluna]))

        return self


    def transform(self, df: pd.DataFrame) -> np.ndarray:
        colunas = []

        for coluna in CATEGORICAS:  
            indices = df[coluna].astype(str).map(self.vocabularios_[coluna]).fillna(0)

            colunas.append(indices.to_numpy(dtype=np.int64))

        return np.column_stack(colunas)
    