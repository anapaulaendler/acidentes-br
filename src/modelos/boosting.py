import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from colunas import CATEGORICAS, NUMERICAS
from features.encoding import _para_texto
from modelos.catalogo import Hiperparametro, ModeloRegistrado, registrar


def _para_float(x):
    return x.astype(float)


def criar_boosting(iteracoes, taxa_aprendizado, profundidade, class_weight):
    # o HistGradientBoosting não aceita a matriz esparsa do one-hot: as categorias viram números inteiros,
    # e o modelo trata cada coluna como categórica (sem ordem). Acima de 255 categorias, as raras viram uma só.
    preparo = ColumnTransformer([
        ("cat", Pipeline([
            ("texto", FunctionTransformer(_para_texto)),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                       encoded_missing_value=np.nan, max_categories=255)),
        ]), list(CATEGORICAS)),
        ("num", FunctionTransformer(_para_float), list(NUMERICAS)),
    ], remainder="drop")

    # boosting: cada árvore nova corrige o erro das anteriores
    modelo = HistGradientBoostingClassifier(
        max_iter=iteracoes,
        learning_rate=taxa_aprendizado,
        max_depth=profundidade,
        class_weight=class_weight,
        categorical_features=[True] * len(CATEGORICAS) + [False] * len(NUMERICAS),
        early_stopping=True,
        random_state=2903,
    )
    return Pipeline([("prep", preparo), ("modelo", modelo)])


registrar(ModeloRegistrado(
    nome="Gradient Boosting",
    construtor=criar_boosting,
    hiperparametros=(
        Hiperparametro("iteracoes", "int", padrao=300, minimo=10, maximo=2000),
        Hiperparametro("taxa_aprendizado", "float", padrao=0.1, minimo=0.001, maximo=1.0),
        Hiperparametro("profundidade", "int", padrao=8, minimo=2, maximo=30),
        Hiperparametro("class_weight", "escolha", padrao="balanced", opcoes=(None, "balanced")),
    ),
    usa_preprocessador=False,
))
