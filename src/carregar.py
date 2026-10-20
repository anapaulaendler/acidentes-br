import pandas as pd

from pathlib import Path
from typing import IO
from validar import validar_colunas
from colunas import ADMINISTRATIVAS

def carregar(fonte: str | Path | IO) -> pd.DataFrame:
    df = pd.read_csv(filepath_or_buffer=fonte, sep=";", encoding="latin1", dtype=str)

    validar_colunas(df)

    df = df.drop(columns=ADMINISTRATIVAS)

    for coluna in ["latitude", "longitude", "km"]:
        df[coluna] = df[coluna].str.replace(",", ".", regex=False)
        df[coluna] = df[coluna].astype(float)

    df["data_inversa"] = pd.to_datetime(df["data_inversa"])

    for coluna in ["br", "pessoas", "veiculos"]:
        df[coluna] = df[coluna].astype("Int64")

    df = df.dropna(subset="classificacao_acidente")

    df["fatal"] = df["classificacao_acidente"] == "Com Vítimas Fatais"

    return df
