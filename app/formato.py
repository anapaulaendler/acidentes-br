import pandas as pd
import plotly.io as pio
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


def grafico(fig, altura: int = 380) -> Markup:
    """Só o pedaço da figura do plotly: a página e o plotly.js já vêm do base.html."""
    fig.update_layout(template="simple_white", height=altura, margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(family="system-ui, sans-serif", size=13))
    return Markup(pio.to_html(fig, full_html=False, include_plotlyjs=False,
                              config={"displayModeBar": False, "responsive": True}))
