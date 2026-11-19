"""A busca de hiperparâmetros (RandomizedSearchCV): os espaços de busca e o arquivo com os resultados.

Os nomes seguem o pipeline do scikit-learn: modelo__max_depth é o max_depth do passo "modelo".
As redes em PyTorch ficam fora: as duas precisam dos mesmos hiperparâmetros para a comparação embeddings × one-hot
(seção 6.9), e cada treino delas leva minutos.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import loguniform, randint

ARQUIVO = Path(__file__).resolve().parents[2] / "modelos_treinados" / "busca.joblib"

# a PR-AUC escolhe a configuração: não depende do limiar, e o limiar é escolhido depois (seções 6.5 e 6.6)
METRICA = "average_precision"

ESPACOS: dict[str, tuple[dict, int]] = {  # nome: (espaço, número de combinações sorteadas)
    "Regressão logística": ({
        "modelo__C": loguniform(0.01, 10),
    }, 8),
    "Árvore de decisão": ({
        "modelo__max_depth": randint(4, 25),
        "modelo__min_samples_leaf": randint(5, 500),
        "modelo__criterion": ["gini", "entropy"],
    }, 20),
    "KNN": ({
        "modelo__n_neighbors": randint(5, 200),
        "modelo__weights": ["uniform", "distance"],
    }, 8),
    "Random Forest": ({
        "modelo__n_estimators": randint(100, 300),
        "modelo__max_depth": randint(8, 25),
        "modelo__min_samples_leaf": randint(5, 100),
        "modelo__max_features": ["sqrt", 0.2, 0.4],
    }, 12),
    "Gradient Boosting": ({
        "modelo__learning_rate": loguniform(0.02, 0.3),
        "modelo__max_depth": randint(3, 12),
        "modelo__max_leaf_nodes": randint(15, 127),
        "modelo__min_samples_leaf": randint(20, 300),
        "modelo__l2_regularization": loguniform(1e-3, 10),
    }, 20),
    "MLP (scikit-learn)": ({
        "modelo__hidden_layer_sizes": [(32,), (64,), (64, 64), (128, 64), (128, 128), (64, 64, 64)],
        "modelo__alpha": loguniform(1e-5, 1e-2),
        "modelo__learning_rate_init": loguniform(1e-4, 1e-2),
    }, 10),
}


@dataclass(frozen=True)
class ResultadoBusca:
    melhores: dict  # a melhor combinação sorteada
    pr_auc_melhores: float
    pr_auc_padrao: float  # os padrões do catálogo, nas mesmas fatias
    escolhidos: dict  # o que o treino usa: a melhor combinação, ou {} se os padrões ganharam
    resultados: pd.DataFrame  # uma linha por combinação sorteada
    acidentes: int
    segundos: float


def salvar(resultados: dict[str, ResultadoBusca]) -> None:
    ARQUIVO.parent.mkdir(exist_ok=True)
    joblib.dump(resultados, ARQUIVO, compress=3)


def carregar() -> dict[str, ResultadoBusca]:
    return joblib.load(ARQUIVO) if ARQUIVO.exists() else {}
