from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class Hiperparametro:
    nome: str
    tipo: Literal["int", "float", "escolha"]
    padrao: int | float | str
    minimo: int | float | None = None
    maximo: int | float | None = None
    opcoes: tuple | None = None


@dataclass(frozen=True)
class ModeloRegistrado:
    nome: str
    construtor: Callable[..., BaseEstimator]
    hiperparametros: tuple[Hiperparametro, ...]


CATALOGO: dict[str, ModeloRegistrado] = {}


def registrar(modelo: ModeloRegistrado) -> None:
    CATALOGO[modelo.nome] = modelo


def criar_modelo(nome: str, **params) -> BaseEstimator:
    modelo = CATALOGO[nome]
    hiperparametros = []

    for hiperparam in modelo.hiperparametros:
        hiperparametros.append(hiperparam.nome)

    desconhecidos = set(params) - set(hiperparametros)
    if desconhecidos:
        raise ValueError(f"Parâmetros desconhecidos para {nome}: {sorted(desconhecidos)}")

    valores = {}
    for hiperparam in modelo.hiperparametros:
        valor = params.get(hiperparam.nome, hiperparam.padrao)

        if hiperparam.tipo == "escolha":
            if valor not in hiperparam.opcoes:
                raise ValueError(f"{hiperparam.nome}={valor!r} fora de {hiperparam.opcoes}")
        else:
            if hiperparam.minimo is not None and valor < hiperparam.minimo:
                raise ValueError(f"{hiperparam.nome}={valor} abaixo do mínimo {hiperparam.minimo}")
            if hiperparam.maximo is not None and valor > hiperparam.maximo:
                raise ValueError(f"{hiperparam.nome}={valor} acima do máximo {hiperparam.maximo}")

        valores[hiperparam.nome] = valor

    return modelo.construtor(**valores)


registrar(ModeloRegistrado(
    nome="Regressão logística",
    construtor=partial(LogisticRegression, max_iter=1000),
    hiperparametros=(
        Hiperparametro("C", "float", padrao=1.0, minimo=0.01, maximo=100),
        Hiperparametro("class_weight", "escolha", padrao="balanced", opcoes=(None, "balanced")),
    ),
))