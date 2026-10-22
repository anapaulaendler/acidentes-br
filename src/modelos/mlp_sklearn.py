from sklearn.neural_network import MLPClassifier
from modelos.catalogo import registrar, ModeloRegistrado, Hiperparametro


def criar_mlp(camadas, neuronios, alpha, taxa_aprendizado):
    return MLPClassifier(
        hidden_layer_sizes=(neuronios,) * camadas,
        alpha=alpha,
        learning_rate_init=taxa_aprendizado,
        early_stopping=True,
        max_iter=200,
        random_state=42,
    )


registrar(ModeloRegistrado(
    nome="MLP (scikit-learn)",
    construtor=criar_mlp,
    hiperparametros=(
        Hiperparametro("camadas", "int", padrao=2, minimo=1, maximo=3),
        Hiperparametro("neuronios", "int", padrao=64, minimo=8, maximo=256),
        Hiperparametro("alpha", "float", padrao=1e-4, minimo=1e-5, maximo=0.1),
        Hiperparametro("taxa_aprendizado", "float", padrao=1e-3, minimo=1e-4, maximo=0.1),
    ),
))
