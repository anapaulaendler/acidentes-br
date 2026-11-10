from sklearn.neighbors import KNeighborsClassifier

from modelos.catalogo import Hiperparametro, ModeloRegistrado, registrar


def criar_knn(vizinhos, pesos):
    return KNeighborsClassifier(n_neighbors=vizinhos, weights=pesos, n_jobs=-1)


registrar(ModeloRegistrado(
    nome="KNN",
    construtor=criar_knn,
    hiperparametros=(
        Hiperparametro("vizinhos", "int", padrao=25, minimo=1, maximo=201),
        Hiperparametro("pesos", "escolha", padrao="uniform", opcoes=("uniform", "distance")),
    )
))