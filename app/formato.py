import pandas as pd
from markupsafe import Markup


def num(valor, casas: int = 1) -> str:
    # 1234.5 → "1.234,5" (formato brasileiro)
    if valor is None or pd.isna(valor):
        return "–"

    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(valor, casas: int = 1) -> str:
    return f"{num(valor, casas)}%"


def tabela(df: pd.DataFrame, casas: int = 2, indice: bool = True) -> Markup:
    formatado = df.copy()

    for coluna in formatado.columns:
        if pd.api.types.is_bool_dtype(formatado[coluna]):
            formatado[coluna] = formatado[coluna].map({True: "sim", False: "não"})
        elif pd.api.types.is_integer_dtype(formatado[coluna]):
            formatado[coluna] = formatado[coluna].map(lambda v: num(v, 0))
        elif pd.api.types.is_float_dtype(formatado[coluna]):
            formatado[coluna] = formatado[coluna].map(lambda v: num(v, casas))

    return Markup(formatado.to_html(classes="tabela", index=indice, border=0, na_rep="–", escape=True))
