import pandas as pd
from sklearn.decomposition import PCA

from features.encoding import DESCONHECIDO


def projetar_2d(embedding: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Reduz o vetor de cada categoria a 2 números (x, y). Devolve também quanto da variação o PCA manteve (%)."""
    categorias = embedding.drop(index=DESCONHECIDO)  # a linha 0 não é uma categoria de verdade
    if categorias.shape[1] < 2:
        raise ValueError("O embedding precisa de pelo menos 2 dimensões.")

    pca = PCA(n_components=2, random_state=2903)
    pontos = pca.fit_transform(categorias)

    projecao = pd.DataFrame(pontos, index=categorias.index, columns=["x", "y"])
    return projecao, pca.explained_variance_ratio_.sum() * 100
