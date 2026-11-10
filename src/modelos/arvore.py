from sklearn.tree import DecisionTreeClassifier

from modelos.catalogo import Hiperparametro, ModeloRegistrado, registrar


def criar_arvore(profundidade, minimo_folha, class_weight):
    return DecisionTreeClassifier(
        max_depth=profundidade,
        min_samples_leaf=minimo_folha,
        class_weight=class_weight,
        random_state=2903
    )


registrar(ModeloRegistrado(
    nome="Árvore de decisão",
    construtor=criar_arvore,
    hiperparametros=(
        Hiperparametro("profundidade", "int", padrao=10, minimo=2, maximo=30),
        Hiperparametro("minimo_folha", "int", padrao=50, minimo=1, maximo=5000),
        Hiperparametro("class_weight", "escolha", padrao="balanced", opcoes=(None, "balanced")),
    ),
))