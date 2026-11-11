import pandas as pd

from colunas import ALVO


def letalidade_por(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    tabela = df.groupby(coluna).agg(acidentes=(ALVO, "size"), fatais=(ALVO, "sum"))
    tabela["letalidade"] = tabela["fatais"] / tabela["acidentes"] * 100

    return tabela

