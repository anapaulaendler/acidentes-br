"""A base completa da PRF (data/raw/) e a amostra de produção, que fica fora do treino."""

from pathlib import Path

import pandas as pd

from carregar import carregar
from limpeza import limpar

RAIZ = Path(__file__).resolve().parent.parent
PASTA_RAW = RAIZ / "data" / "raw"
# os acidentes que simulam dados novos: ficam fora do treino e da validação
PRODUCAO = RAIZ / "data" / "amostras" / "producao_100.csv"
ANOS = range(2017, 2027)
NOME = f"Base completa PRF {ANOS[0]}–{ANOS[-1]}"


def _limpar(bruto: pd.DataFrame) -> pd.DataFrame:
    limpo = limpar(bruto)
    limpo["ano"] = limpo["data_inversa"].dt.year
    return limpo


def _sem_producao(bruto: pd.DataFrame) -> pd.DataFrame:
    if not PRODUCAO.exists():
        return bruto

    producao = carregar(PRODUCAO).drop_duplicates()
    # o merge compara todas as colunas; o indicator diz se a linha também está na produção
    marcado = bruto.merge(producao, how="left", indicator=True)
    return bruto[(marcado["_merge"] == "left_only").to_numpy()]


def base_de_treino() -> pd.DataFrame:
    """A base completa, limpa e sem os acidentes da amostra de produção."""
    faltando = [ano for ano in ANOS if not (PASTA_RAW / f"datatran{ano}.csv").exists()]
    if faltando:
        raise FileNotFoundError(f"Faltam em {PASTA_RAW}: " + ", ".join(f"datatran{ano}.csv" for ano in faltando))

    bruto = pd.concat([carregar(PASTA_RAW / f"datatran{ano}.csv") for ano in ANOS], ignore_index=True)
    return _limpar(_sem_producao(bruto))


def producao() -> pd.DataFrame:
    """A amostra de produção, limpa: acidentes que os modelos nunca viram."""
    return _limpar(carregar(PRODUCAO))
