import pandas as pd

from sklearn.pipeline import Pipeline
from features.encoding import preprocessador
from modelos.catalogo import CATALOGO, criar_modelo
from modelos.personas import MarcarPersona
from colunas import ALVO


def montar(nome: str, ajustes: dict | None = None, **params) -> Pipeline:
    """O pipeline do modelo, sem treinar: a persona do trecho, o preparo das colunas e o modelo.

    params usa os nomes do catálogo (profundidade, ...). ajustes usa os nomes do scikit-learn no pipeline
    (modelo__max_depth, ...), os mesmos da busca de hiperparâmetros.
    """
    registrado = CATALOGO[nome]
    modelo = criar_modelo(nome, **params)

    if not registrado.usa_preprocessador:
        # o Gradient Boosting já é um pipeline com o preparo dele; as redes em PyTorch preparam as colunas no fit
        passos = modelo.steps if isinstance(modelo, Pipeline) else [("modelo", modelo)]
    else:
        passos = [("prep", preprocessador()), ("modelo", modelo)]

    pipeline = Pipeline([("persona", MarcarPersona())] + passos)
    if ajustes:
        pipeline.set_params(**ajustes)
    return pipeline


def treinar(nome: str, df: pd.DataFrame, ajustes: dict | None = None, **params) -> Pipeline:
    pipeline = montar(nome, ajustes, **params)
    pipeline.fit(df, df[ALVO])
    return pipeline
