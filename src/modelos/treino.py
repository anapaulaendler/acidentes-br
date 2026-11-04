import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from features.encoding import preprocessador
from modelos.catalogo import CATALOGO, criar_modelo
from colunas import ALVO


def treinar(nome: str, df: pd.DataFrame, **params) -> BaseEstimator:
    modelo = criar_modelo(nome, **params)

    if not CATALOGO[nome].usa_preprocessador:
        modelo.fit(df, df[ALVO]) # type: ignore
        return modelo
    
    pipeline = Pipeline([
        ("prep", preprocessador()),
        ("modelo", criar_modelo(nome, **params))
    ])

    pipeline.fit(df, df[ALVO])

    return pipeline
