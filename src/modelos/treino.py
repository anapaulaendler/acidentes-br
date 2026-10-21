import pandas as pd

from sklearn.pipeline import Pipeline
from features.encoding import preprocessador
from modelos.catalogo import criar_modelo
from colunas import ALVO


def treinar(nome: str, df: pd.DataFrame, **params) -> Pipeline:
    pipeline = Pipeline([
        ("prep", preprocessador()),
        ("modelo", criar_modelo(nome, **params))
    ])

    pipeline.fit(df, df[ALVO])

    return pipeline
