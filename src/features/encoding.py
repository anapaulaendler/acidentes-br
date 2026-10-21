from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from colunas import CATEGORICAS, NUMERICAS


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
