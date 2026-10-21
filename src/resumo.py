import pandas as pd

from colunas import ALVO


def resumo_base(df: pd.DataFrame) -> dict:
    return {
        "linhas": len(df),
        "pct_fatal": float(df[ALVO].mean() * 100),
        "anos": sorted(int(ano) for ano in df["data_inversa"].dt.year.dropna().unique()),
        "ufs": sorted(df["uf"].dropna().unique().tolist()),
    }
