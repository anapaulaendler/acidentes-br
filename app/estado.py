"""A base da app: a base inicial mais os arquivos adicionados, que ficam salvos em data/adicionados/."""

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from werkzeug.utils import secure_filename

from analises.comparacao import comparar_modelos, comparar_perfil
from carregar import carregar
from limpeza import limpar
from modelos import treinados

RAIZ = Path(__file__).resolve().parent.parent
PASTA_RAW = RAIZ / "data" / "raw"
PASTA_ADICIONADOS = RAIZ / "data" / "adicionados"
AMOSTRA = RAIZ / "data" / "amostras" / "acidentes_pr_2024.csv"
ANOS_BASE_COMPLETA = range(2017, 2027)


@dataclass(frozen=True)
class Base:
    nome: str
    bruto: pd.DataFrame
    limpo: pd.DataFrame
    versao: int
    arquivos_adicionados: int


@dataclass(frozen=True)
class Comparacao:
    arquivo: str
    linhas_arquivo: int
    linhas_novas: int  # as que ainda não estavam na base
    perfil: pd.DataFrame
    modelo: pd.DataFrame | None


_base: Base | None = None
_modelos: treinados.ModelosTreinados | None = None
_ultima_comparacao: Comparacao | None = None
_versao = 0
_cache: dict[tuple, Any] = {}


def base() -> Base | None:
    return _base


def modelos() -> treinados.ModelosTreinados | None:
    return _modelos


def ultima_comparacao() -> Comparacao | None:
    return _ultima_comparacao


def _limpar(bruto: pd.DataFrame) -> pd.DataFrame:
    limpo = limpar(bruto)
    limpo["ano"] = limpo["data_inversa"].dt.year
    return limpo


def _guardar(nome: str, bruto: pd.DataFrame, arquivos_adicionados: int, limpo: pd.DataFrame | None = None) -> None:
    global _base, _versao

    _versao += 1
    limpo = _limpar(bruto) if limpo is None else limpo
    _base = Base(nome=nome, bruto=bruto, limpo=limpo, versao=_versao, arquivos_adicionados=arquivos_adicionados)
    _cache.clear()


def tem_base_completa() -> bool:
    return all((PASTA_RAW / f"datatran{ano}.csv").exists() for ano in ANOS_BASE_COMPLETA)


def _base_inicial() -> tuple[str, pd.DataFrame]:
    if tem_base_completa():
        partes = [carregar(PASTA_RAW / f"datatran{ano}.csv") for ano in ANOS_BASE_COMPLETA]
        return f"Base completa PRF {ANOS_BASE_COMPLETA[0]}–{ANOS_BASE_COMPLETA[-1]}", pd.concat(partes, ignore_index=True)

    return "Amostra PR 2024", carregar(AMOSTRA)


def arquivos_adicionados() -> list[Path]:
    if not PASTA_ADICIONADOS.exists():
        return []

    return sorted(PASTA_ADICIONADOS.iterdir())


def iniciar(com_modelos: bool = True) -> Base:
    """Monta a base inicial com tudo o que já foi adicionado, e carrega os modelos treinados."""
    global _modelos, _ultima_comparacao

    nome, bruto = _base_inicial()
    adicionados = arquivos_adicionados()
    partes = [bruto] + [carregar(arquivo) for arquivo in adicionados]
    _guardar(nome, pd.concat(partes, ignore_index=True), len(adicionados))

    _modelos = treinados.carregar() if com_modelos else None
    _ultima_comparacao = None
    return _base


def adicionar(conteudo: bytes, nome_arquivo: str, formato: str) -> Comparacao:
    """Lê o arquivo, junta à base e salva uma cópia em data/adicionados/ para a junção valer nas próximas execuções."""
    global _ultima_comparacao

    # se o arquivo for inválido, o erro sobe aqui, antes de salvar qualquer coisa
    novos = carregar(io.BytesIO(conteudo), formato)
    anterior = _base

    bruto = pd.concat([anterior.bruto, novos], ignore_index=True)
    limpo = _limpar(bruto)
    linhas_novas = len(limpo) - len(anterior.limpo)

    # um arquivo que só repete a base não muda nada e não é salvo
    if linhas_novas:
        PASTA_ADICIONADOS.mkdir(parents=True, exist_ok=True)
        destino = PASTA_ADICIONADOS / f"{datetime.now():%Y%m%d-%H%M%S}_{secure_filename(nome_arquivo)}"
        destino.write_bytes(conteudo)
        _guardar(anterior.nome, bruto, anterior.arquivos_adicionados + 1, limpo)

    novos_limpos = _limpar(novos)
    _ultima_comparacao = Comparacao(
        arquivo=nome_arquivo,
        linhas_arquivo=len(novos_limpos),
        linhas_novas=linhas_novas,
        perfil=comparar_perfil(anterior.limpo, novos_limpos),
        modelo=comparar_modelos(_modelos.classificadores, novos_limpos) if _modelos else None,
    )
    return _ultima_comparacao


def remover_adicionados() -> None:
    for arquivo in arquivos_adicionados():
        arquivo.unlink()

    iniciar()


def em_cache(chave: tuple, calcular: Callable[[], Any]) -> Any:
    chave = (_versao,) + chave
    if chave not in _cache:
        _cache[chave] = calcular()

    return _cache[chave]
