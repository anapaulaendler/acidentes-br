import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset

from features.encoding import CodificadorCategorias
from colunas import CATEGORICAS, NUMERICAS

import torch
import torch.nn as nn


# cardinalidade = quantas categorias a coluna tem (inclui "<desconhecido>");
# devolve quantas dimensões o vetor de cada categoria vai ter
def tamanho_embedding(cardinalidade: int) -> int:
    return min(50, (cardinalidade + 1) // 2)


class RedeEmbeddings(nn.Module):
    def __init__(self, cardinalidades: list[int], n_numericas: int, camadas: int, neuronios: int, dropout: float) -> None:
        super().__init__()

        self.embeddings = nn.ModuleList()
        for cardinalidade in cardinalidades:
            self.embeddings.append(nn.Embedding(cardinalidade, tamanho_embedding(cardinalidade)))

        entrada = sum(tamanho_embedding(cardinalidade) for cardinalidade in cardinalidades) + n_numericas

        blocos = []
        for _ in range(camadas):
            blocos.append(nn.Linear(entrada, neuronios))
            blocos.append(nn.ReLU()) # transforma valores negativos em 0 
            blocos.append(nn.Dropout(dropout))
            
            entrada = neuronios

        blocos.append(nn.Linear(neuronios, 1))

        self.camadas = nn.Sequential(*blocos) 


    def forward(self, x_categoricas: torch.LongTensor, x_numericas: torch.FloatTensor) -> torch.Tensor:
        vetores = []
        for i, embedding in enumerate(self.embeddings):
            coluna = x_categoricas[:, i]
            
            vetores.append(embedding(coluna))

        x = torch.cat(vetores + [x_numericas], dim=1)

        pontuacoes = self.camadas(x)

        return pontuacoes.squeeze(-1)


class ClassificadorEmbeddings(ClassifierMixin, BaseEstimator):
    def __init__(self, epocas: int = 20, tamanho_lote: int = 512, taxa_aprendizado: float = 0.001,
                 camadas: int = 2, neuronios: int = 64, dropout: float = 0.2, 
                 class_weight: str | None = "balanced", random_state: int = 2903) -> None:
        self.epocas = epocas
        self.tamanho_lote = tamanho_lote
        self.taxa_aprendizado = taxa_aprendizado
        self.camadas = camadas
        self.neuronios = neuronios
        self.dropout = dropout
        self.class_weight = class_weight
        self.random_state = random_state


    def fit(self, X: pd.DataFrame, y) -> "ClassificadorEmbeddings":
        self.codificador_ = CodificadorCategorias().fit(X)

        self.numericas_ = Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("padronizar", StandardScaler())
        ]).fit(X[list(NUMERICAS)])

        x_cat = torch.tensor(self.codificador_.transform(X), dtype=torch.long)
        x_num = torch.tensor(self.numericas_.transform(X[list(NUMERICAS)]), dtype=torch.float32)

        alvo = torch.tensor(np.asarray(y), dtype=torch.float32)

        torch.manual_seed(self.random_state)

        self.rede_ = RedeEmbeddings(self.codificador_.cardinalidades_, len(NUMERICAS), self.camadas, self.neuronios, self.dropout)

        pos_weight = None
        if self.class_weight == "balanced":
            n_positivos = alvo.sum()
            n_negativos = len(alvo) - n_positivos
            pos_weight = n_negativos / n_positivos

        funcao_perda = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        otimizador = torch.optim.Adam(self.rede_.parameters(), lr=self.taxa_aprendizado)

        lotes = DataLoader(TensorDataset(x_cat, x_num, alvo), batch_size=self.tamanho_lote, shuffle=True)

        self.historico_perda_ = []
        for _ in range(self.epocas):
            self.rede_.train()

            soma=0.0
            for lote_cat, lote_num, lote_alvo in lotes:
                otimizador.zero_grad()

                pontuacoes = self.rede_(lote_cat, lote_num) # vai chamar internamente a função forward, ali de cima
                perda = funcao_perda(pontuacoes, lote_alvo)

                perda.backward() # calcula os gradientes

                otimizador.step()

                soma += perda.item() * len(lote_alvo)

            self.historico_perda_.append(soma / len(alvo))

        self.classes_ = np.array([False, True]) # nosso resultado previsto

        self.embeddings_ = {}
        for coluna, embedding in zip(CATEGORICAS, self.rede_.embeddings):
            vocabulario = self.codificador_.vocabularios_[coluna] # { categoria: indice }
            categorias = sorted(vocabulario, key=vocabulario.get)

            pesos = embedding.weight.detach().numpy() # separa os pesos do grafo pra gerar tabela de embeddings_ 

            self.embeddings_[coluna] = pd.DataFrame(pesos, index=categorias)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        x_cat = torch.tensor(self.codificador_.transform(X), dtype=torch.long)
        x_num = torch.tensor(self.numericas_.transform(X[list(NUMERICAS)]), dtype=torch.float32)

        self.rede_.eval()

        with torch.no_grad():
            pontuacoes = self.rede_(x_cat, x_num)

        p = torch.sigmoid(pontuacoes).numpy()

        return np.column_stack([1 - p, p])


from modelos.catalogo import Hiperparametro, ModeloRegistrado, registrar


registrar(ModeloRegistrado(
    nome="Rede com embeddings (PyTorch)",
    construtor=ClassificadorEmbeddings,
    hiperparametros=(
        Hiperparametro("epocas", "int", padrao=20, minimo=1, maximo=100),
        Hiperparametro("tamanho_lote", "escolha", padrao=512, opcoes=(128, 256, 512, 1024)),
        Hiperparametro("taxa_aprendizado", "float", padrao=0.001, minimo=0.0001, maximo=0.1),
        Hiperparametro("camadas", "int", padrao=2, minimo=1, maximo=3),
        Hiperparametro("neuronios", "int", padrao=64, minimo=8, maximo=256),
        Hiperparametro("dropout", "float", padrao=0.2, minimo=0.0, maximo=0.5),
        Hiperparametro("class_weight", "escolha", padrao="balanced", opcoes=(None, "balanced")),
    ),
    usa_preprocessador=False,
))