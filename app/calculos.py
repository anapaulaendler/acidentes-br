"""Os cálculos de src/ com o cache da base atual: cada resultado é calculado uma vez por base e por filtro."""

import pandas as pd

from app import estado
from app.filtros import Filtros
from modelos import personas, regressao_trechos
from modelos.classificacao import avaliar_classificadores
from trechos import montar_trechos

CLASSIFICADORES = ["Regressão logística", "MLP (scikit-learn)"]


def trechos(df: pd.DataFrame, selecao: Filtros) -> pd.DataFrame:
    return estado.em_cache(("trechos", selecao.chave()), lambda: montar_trechos(df))


def trechos_validos(df: pd.DataFrame, selecao: Filtros) -> int:
    return int((trechos(df, selecao)["acidentes"] >= regressao_trechos.MINIMO_ACIDENTES).sum())


def regressao(df: pd.DataFrame, selecao: Filtros) -> regressao_trechos.ResultadoRegressao | None:
    return estado.em_cache(("regressao", selecao.chave()), lambda: regressao_trechos.ajustar(trechos(df, selecao)))


def agrupamento(df: pd.DataFrame, selecao: Filtros) -> pd.DataFrame | None:
    return estado.em_cache(("personas", selecao.chave()), lambda: personas.agrupar(trechos(df, selecao)))


def diagnostico_k(df: pd.DataFrame, selecao: Filtros) -> pd.DataFrame | None:
    return estado.em_cache(("diagnostico_k", selecao.chave()), lambda: personas.diagnostico_k(trechos(df, selecao)))


def classificacao(df: pd.DataFrame, selecao: Filtros):
    return estado.em_cache(("classificacao", selecao.chave()), lambda: avaliar_classificadores(df, CLASSIFICADORES))
