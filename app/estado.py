from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable

import pandas as pd

from carregar import carregar
from limpeza import limpar

RAIZ = Path(__file__).resolve().parent.parent
PASTA_RAW = RAIZ / "data" / "raw"
AMOSTRA = RAIZ / "data" / "amostras" / "acidentes_pr_2024.csv"
ANOS_BASE_COMPLETA = range(2017, 2026)


@dataclass(frozen=True)
class Base:
    nome: str
    bruto: pd.DataFrame
    limpo: pd.DataFrame
    versao: int


_base: Base | None = None
_versao = 0
_cache: dict[tuple, Any] = {}


def base() -> Base | None:
    return _base


def _guardar(nome: str, bruto: pd.DataFrame) -> None:
    global _base, _versao

    limpo = limpar(bruto)
    limpo["ano"] = limpo["data_inversa"].dt.year

    _versao += 1
    _base = Base(nome=nome, bruto=bruto, limpo=limpo, versao=_versao)
    _cache.clear()


def carregar_base(fonte: str | Path | IO, nome: str, formato: str | None = None) -> None:
    _guardar(nome, carregar(fonte, formato))


def tem_base_completa() -> bool:
    return all((PASTA_RAW / f"datatran{ano}.csv").exists() for ano in ANOS_BASE_COMPLETA)


def carregar_base_completa() -> None:
    partes = [carregar(PASTA_RAW / f"datatran{ano}.csv") for ano in ANOS_BASE_COMPLETA]
    _guardar("Base completa PRF 2017–2025", pd.concat(partes, ignore_index=True))


def em_cache(chave: tuple, calcular: Callable[[], Any]) -> Any:
    chave = (_versao,) + chave
    if chave not in _cache:
        _cache[chave] = calcular()

    return _cache[chave]


def limpar_estado() -> None:
    global _base
    _base = None
    _cache.clear()
