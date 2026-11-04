import pandas as pd

from limpeza import LIMITES_BRASIL


def _fora_do_brasil(df: pd.DataFrame) -> pd.Series:
    fora = pd.Series(False, index=df.index)
    for coluna, (minimo, maximo) in LIMITES_BRASIL.items():
        fora |= (df[coluna] < minimo) | (df[coluna] > maximo)

    return fora


def _contagens(df: pd.DataFrame, coluna_tipo: str) -> list[int]:
    return [
        len(df),
        int(df.duplicated().sum()),
        int((df["condicao_metereologica"] == "Ignorado").sum()),
        int((df["sentido_via"] == "Não Informado").sum()),
        int((df["br"] == 0).sum()),
        df["tracado_via"].nunique(),
        df["causa_acidente"].nunique(),
        df[coluna_tipo].nunique(),
        int(_fora_do_brasil(df).sum()),
    ]


def relatorio_limpeza(bruto: pd.DataFrame, limpo: pd.DataFrame) -> pd.DataFrame:
    # a mesma tabela da seção 3.7 do notebook
    return pd.DataFrame({
        "antes": _contagens(bruto, "tipo_acidente"),
        "depois": _contagens(limpo, "grupo_tipo"),
    }, index=[
        "linhas",
        "linhas repetidas",
        "clima 'Ignorado'",
        "sentido 'Não Informado'",
        "br = 0",
        "textos diferentes em tracado_via",
        "causas diferentes",
        "tipos de acidente (depois: grupo_tipo)",
        "coordenadas fora do Brasil",
    ])


def causas_por_ano(df: pd.DataFrame) -> pd.Series:
    return df.groupby(df["data_inversa"].dt.year)["causa_acidente"].nunique()


def proporcao_grupos_por_ano(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["grupo_causa"], df["data_inversa"].dt.year, normalize="columns") * 100
