import pandas as pd

from colunas import ALVO


def letalidade_por(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    tabela = df.groupby(coluna).agg(acidentes=(ALVO, "size"), fatais=(ALVO, "sum"))
    tabela["letalidade"] = tabela["fatais"] / tabela["acidentes"] * 100

    return tabela


def frontal_por_pista(df: pd.DataFrame) -> pd.DataFrame:
    frontal = df["tipo_acidente"] == "Colisão frontal"

    return pd.DataFrame({
        "pct_frontal": frontal.groupby(df["tipo_pista"]).mean() * 100,
        "letalidade_frontal": df[frontal].groupby("tipo_pista")[ALVO].mean() * 100,
    })


def letalidade_por_ano_e_pista(df: pd.DataFrame) -> pd.DataFrame:
    ano = df["data_inversa"].dt.year.rename("ano")

    return df.pivot_table(index=ano, columns="tipo_pista", values=ALVO, aggfunc="mean") * 100
