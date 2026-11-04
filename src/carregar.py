import io

import pandas as pd

from pathlib import Path
from typing import IO
from validar import ErroDeFormato, validar_colunas
from colunas import ADMINISTRATIVAS

FORMATOS = ("csv", "json", "xml")


def _formato_pelo_nome(fonte: str | Path | IO) -> str:
    nome = getattr(fonte, "filename", None) or (fonte if isinstance(fonte, (str, Path)) else "")
    return Path(nome).suffix.lower().lstrip(".")


def _texto_do_csv(fonte: str | Path | IO) -> str:
    conteudo = Path(fonte).read_bytes() if isinstance(fonte, (str, Path)) else fonte.read()
    if isinstance(conteudo, str):
        return conteudo

    try:
        return conteudo.decode("utf-8")
    except UnicodeDecodeError:
        return conteudo.decode("latin1")


def _ler(fonte: str | Path | IO, formato: str) -> pd.DataFrame:
    if formato == "csv":
        return pd.read_csv(io.StringIO(_texto_do_csv(fonte)), sep=";", dtype=str)

    if formato == "json":
        df = pd.read_json(fonte, orient="records", dtype=False)
    else:
        df = pd.read_xml(fonte, parser="etree", dtype=str)

    return df.astype("str").where(df.notna())


def carregar(fonte: str | Path | IO, formato: str | None = None) -> pd.DataFrame:
    formato = formato or _formato_pelo_nome(fonte)
    if formato not in FORMATOS:
        raise ErroDeFormato(formato)

    df = _ler(fonte, formato)

    validar_colunas(df)

    df = df.drop(columns=list(ADMINISTRATIVAS))

    for coluna in ["latitude", "longitude", "km"]:
        df[coluna] = df[coluna].str.replace(",", ".", regex=False)
        df[coluna] = df[coluna].astype(float)

    df["data_inversa"] = pd.to_datetime(df["data_inversa"])

    for coluna in ["br", "pessoas", "veiculos"]:
        df[coluna] = df[coluna].astype("Int64")

    df = df.dropna(subset="classificacao_acidente")

    df["fatal"] = df["classificacao_acidente"] == "Com Vítimas Fatais"

    return df
