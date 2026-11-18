from sklearn.ensemble import RandomForestClassifier

from modelos.catalogo import Hiperparametro, ModeloRegistrado, registrar


def criar_floresta(arvores, profundidade, minimo_folha, class_weight):
    # bagging: cada árvore treina com uma amostra sorteada dos acidentes, e as árvores votam juntas
    return RandomForestClassifier(
        n_estimators=arvores,
        max_depth=profundidade,
        min_samples_leaf=minimo_folha,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=2903,
    )


registrar(ModeloRegistrado(
    nome="Random Forest",
    construtor=criar_floresta,
    hiperparametros=(
        Hiperparametro("arvores", "int", padrao=100, minimo=10, maximo=500),
        Hiperparametro("profundidade", "int", padrao=15, minimo=2, maximo=40),
        Hiperparametro("minimo_folha", "int", padrao=20, minimo=1, maximo=5000),
        Hiperparametro("class_weight", "escolha", padrao="balanced", opcoes=(None, "balanced", "balanced_subsample")),
    ),
))
