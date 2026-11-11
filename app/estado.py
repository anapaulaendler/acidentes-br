"""A base da app: a base inicial mais os arquivos adicionados, que ficam salvos em data/adicionados/."""

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from werkzeug.utils import secure_filename

from avaliacao import metricas, montar_features, validacao_cruzada
from modelos.treino import treinar
from carregar import carregar
from limpeza import limpar
from colunas import ALVO
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
class Envio:
    """O resultado de um arquivo enviado, para o aviso da tela."""
    arquivo: str
    linhas_arquivo: int
    linhas_novas: int  # as que ainda não estavam na base


@dataclass(frozen=True)
class Previsao:
    """O que os modelos treinados dizem sobre os dados adicionados."""
    metricas: pd.DataFrame  # uma linha por modelo
    previstos: pd.DataFrame  # os dados adicionados, com uma coluna por modelo (True = o modelo previu fatal)
    feita_em: datetime


LIMIAR = 0.5  # o mesmo limiar da validação cruzada

_base: Base | None = None
_modelos: treinados.ModelosTreinados | None = None
_adicionados: pd.DataFrame | None = None  # calculado na primeira vez que a tela pede
_previsao: Previsao | None = None


def base() -> Base | None:
    return _base


def modelos() -> treinados.ModelosTreinados | None:
    return _modelos


def dados_adicionados() -> pd.DataFrame:
    """As linhas dos arquivos enviados pela app, limpas, com o nome do arquivo de onde vieram."""
    global _adicionados

    if _adicionados is None:
        partes = []
        for arquivo in arquivos_adicionados():
            # o nome salvo é "AAAAMMDD-HHMMSS_nome.csv"; a tela mostra só o nome enviado
            partes.append(_limpar(carregar(arquivo)).assign(arquivo=arquivo.name.split("_", 1)[-1]))
        _adicionados = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()

    return _adicionados


def previsao() -> Previsao | None:
    return _previsao


def prever() -> Previsao | None:
    """Roda cada modelo treinado nos dados adicionados e guarda as previsões e as métricas."""
    global _previsao

    dados = dados_adicionados()
    if dados.empty or _modelos is None:
        return None

    features = montar_features(dados)
    previstos = dados.copy()
    linhas = {}
    for nome, classificador in _modelos.classificadores.items():
        probabilidades = classificador.modelo.predict_proba(features)[:, 1]
        previstos[nome] = probabilidades >= LIMIAR
        linhas[nome] = metricas(dados[ALVO], probabilidades, LIMIAR)

    _previsao = Previsao(metricas=pd.DataFrame.from_dict(linhas, orient="index"), previstos=previstos, feita_em=datetime.now())
    return _previsao


def retreinar(nome: str, params: dict, fatias: int) -> treinados.Classificador:
    """Treina um modelo de novo com a base atual (inicial + adicionados) e os hiperparâmetros da tela, e salva.

    É a alternativa pesada ao prever(): o modelo passa a conhecer os dados novos.
    Os outros modelos continuam como estavam.
    """
    global _modelos, _previsao

    # como no script de treino: a amostra de produção nunca entra no treino
    df = _limpar(_sem_producao(_base.bruto))
    por_fatia = validacao_cruzada(nome, df, fatias, **params)
    classificador = treinados.Classificador(
        modelo=treinar(nome, montar_features(df), **params), metricas_fatias=por_fatia, parametros=params,
    )

    anteriores = _modelos.classificadores if _modelos else {}
    novos = {**anteriores, nome: classificador}
    _modelos = treinados.ModelosTreinados(
        treinado_em=datetime.now(),
        base=_base.nome,
        acidentes=len(df),
        anos=f"{df['ano'].min()}–{df['ano'].max()}",
        classificadores={n: novos[n] for n in treinados.CLASSIFICADORES if n in novos},
    )
    treinados.salvar(_modelos)
    _previsao = None  # a previsão antiga é de outro modelo
    return classificador


def _limpar(bruto: pd.DataFrame) -> pd.DataFrame:
    limpo = limpar(bruto)
    limpo["ano"] = limpo["data_inversa"].dt.year
    return limpo


def _guardar(nome: str, bruto: pd.DataFrame, arquivos_adicionados: int, limpo: pd.DataFrame | None = None) -> None:
    global _base, _adicionados, _previsao

    _adicionados = None
    _previsao = None
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
    global _base, _modelos, _adicionados, _previsao

    _adicionados = None
    _previsao = None
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
    return _base


def adicionar(conteudo: bytes, nome_arquivo: str, formato: str) -> Envio:
    """Lê o arquivo, junta à base e salva uma cópia em data/adicionados/ para a junção valer nas próximas execuções."""
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

    return Envio(arquivo=nome_arquivo, linhas_arquivo=len(_limpar(novos)), linhas_novas=linhas_novas)


def _primeira_base(conteudo: bytes, nome_arquivo: str, novos: pd.DataFrame) -> Envio:
    """Sem base inicial, o primeiro arquivo enviado vira a base."""
    _salvar_adicionado(conteudo, nome_arquivo)
    _guardar("Arquivos enviados", novos, 1)
    linhas = len(_base.limpo)
    return Envio(arquivo=nome_arquivo, linhas_arquivo=linhas, linhas_novas=linhas)


def _salvar_adicionado(conteudo: bytes, nome_arquivo: str) -> None:
    PASTA_ADICIONADOS.mkdir(parents=True, exist_ok=True)
    destino = PASTA_ADICIONADOS / f"{datetime.now():%Y%m%d-%H%M%S}_{secure_filename(nome_arquivo)}"
    destino.write_bytes(conteudo)


def remover_adicionados() -> None:
    for arquivo in arquivos_adicionados():
        arquivo.unlink()

    iniciar()
