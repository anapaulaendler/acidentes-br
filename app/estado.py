"""A base da app: a base inicial mais os arquivos adicionados, que ficam salvos em data/adicionados/."""

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from werkzeug.utils import secure_filename

from analises.comparacao import comparar_modelos, comparar_perfil
from carregar import carregar
from limpeza import limpar
from modelos import treinados

RAIZ = Path(__file__).resolve().parent.parent
PASTA_RAW = RAIZ / "data" / "raw"
PASTA_ADICIONADOS = RAIZ / "data" / "adicionados"
# os acidentes da demonstração de "produção": ficam fora da base de treino
PRODUCAO = RAIZ / "data" / "amostras" / "producao_100.csv"
ANOS_BASE_COMPLETA = range(2017, 2027)


@dataclass(frozen=True)
class Base:
    nome: str
    bruto: pd.DataFrame
    limpo: pd.DataFrame
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
    global _base

    limpo = _limpar(bruto) if limpo is None else limpo
    _base = Base(nome=nome, bruto=bruto, limpo=limpo, arquivos_adicionados=arquivos_adicionados)


def tem_base_completa() -> bool:
    return all((PASTA_RAW / f"datatran{ano}.csv").exists() for ano in ANOS_BASE_COMPLETA)


def _sem_producao(bruto: pd.DataFrame) -> pd.DataFrame:
    """Tira da base os acidentes da amostra de produção: o treino nunca vê esses acidentes."""
    if not PRODUCAO.exists():
        return bruto

    producao = carregar(PRODUCAO).drop_duplicates()
    # o merge compara todas as colunas; o indicator diz se a linha também está na produção
    marcado = bruto.merge(producao, how="left", indicator=True)
    return bruto[(marcado["_merge"] == "left_only").to_numpy()]


def _base_inicial() -> tuple[str, pd.DataFrame | None]:
    """A base completa da PRF, se os arquivos estão em data/raw/; sem eles, a base começa só com os arquivos enviados."""
    if not tem_base_completa():
        return "Arquivos enviados", None

    partes = [carregar(PASTA_RAW / f"datatran{ano}.csv") for ano in ANOS_BASE_COMPLETA]
    return f"Base completa PRF {ANOS_BASE_COMPLETA[0]}–{ANOS_BASE_COMPLETA[-1]}", _sem_producao(pd.concat(partes, ignore_index=True))


def arquivos_adicionados() -> list[Path]:
    if not PASTA_ADICIONADOS.exists():
        return []

    return sorted(PASTA_ADICIONADOS.iterdir())


def iniciar(com_modelos: bool = True, para_treino: bool = False) -> Base | None:
    """Monta a base inicial com tudo o que já foi adicionado, e carrega os modelos treinados.

    Com para_treino=True, a amostra de produção sai também dos arquivos adicionados: se alguém
    enviou o producao_100.csv pela app, o treino e a validação cruzada continuam sem esses acidentes.
    """
    global _base, _modelos, _ultima_comparacao

    nome, inicial = _base_inicial()
    adicionados = arquivos_adicionados()
    partes = ([] if inicial is None else [inicial]) + [carregar(arquivo) for arquivo in adicionados]
    if partes:
        bruto = pd.concat(partes, ignore_index=True)
        if para_treino:
            bruto = _sem_producao(bruto)
        _guardar(nome, bruto, len(adicionados))
    else:
        # sem base: as telas mandam para a tela Dados, onde o primeiro arquivo enviado vira a base
        _base = None

    _modelos = treinados.carregar() if com_modelos else None
    _ultima_comparacao = None
    return _base


def adicionar(conteudo: bytes, nome_arquivo: str, formato: str) -> Comparacao:
    """Lê o arquivo, junta à base e salva uma cópia em data/adicionados/ para a junção valer nas próximas execuções."""
    global _ultima_comparacao

    # se o arquivo for inválido, o erro sobe aqui, antes de salvar qualquer coisa
    novos = carregar(io.BytesIO(conteudo), formato)
    anterior = _base
    if anterior is None:
        return _primeira_base(conteudo, nome_arquivo, novos)

    bruto = pd.concat([anterior.bruto, novos], ignore_index=True)
    limpo = _limpar(bruto)
    linhas_novas = len(limpo) - len(anterior.limpo)

    # um arquivo que só repete a base não muda nada e não é salvo
    if linhas_novas:
        _salvar_adicionado(conteudo, nome_arquivo)
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


def _primeira_base(conteudo: bytes, nome_arquivo: str, novos: pd.DataFrame) -> Comparacao:
    """Sem base inicial, o primeiro arquivo enviado vira a base: não há com o que comparar."""
    global _ultima_comparacao

    _salvar_adicionado(conteudo, nome_arquivo)
    _guardar("Arquivos enviados", novos, 1)
    _ultima_comparacao = None
    linhas = len(_base.limpo)
    return Comparacao(arquivo=nome_arquivo, linhas_arquivo=linhas, linhas_novas=linhas, perfil=pd.DataFrame(), modelo=None)


def _salvar_adicionado(conteudo: bytes, nome_arquivo: str) -> None:
    PASTA_ADICIONADOS.mkdir(parents=True, exist_ok=True)
    destino = PASTA_ADICIONADOS / f"{datetime.now():%Y%m%d-%H%M%S}_{secure_filename(nome_arquivo)}"
    destino.write_bytes(conteudo)


def remover_adicionados() -> None:
    for arquivo in arquivos_adicionados():
        arquivo.unlink()

    iniciar()
