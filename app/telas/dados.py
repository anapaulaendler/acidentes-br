from dataclasses import dataclass

import pandas as pd
from flask import Blueprint, render_template, request

from app.telas import com_base_filtrada

bp = Blueprint("dados", __name__, url_prefix="/dados")

LINHAS_POR_PAGINA = 50

COLUNAS_TABELA = ["data_inversa", "horario", "uf", "br", "km", "municipio", "causa_acidente", "grupo_causa",
                  "tipo_acidente", "tipo_pista", "condicao_metereologica", "fatal"]


@dataclass(frozen=True)
class Pagina:
    tabela: pd.DataFrame
    total: int
    pagina: int
    paginas: int


def paginar(df: pd.DataFrame, colunas: list[str]) -> Pagina:
    """Corta a página pedida na URL (?pagina=N) de uma tabela já filtrada."""
    paginas = max(1, -(-len(df) // LINHAS_POR_PAGINA))
    pagina = min(max(request.args.get("pagina", 1, type=int), 1), paginas)

    inicio = (pagina - 1) * LINHAS_POR_PAGINA
    tabela = df[colunas].iloc[inicio:inicio + LINHAS_POR_PAGINA].copy()
    tabela["data_inversa"] = tabela["data_inversa"].dt.strftime("%d/%m/%Y")
    return Pagina(tabela=tabela, total=len(df), pagina=pagina, paginas=paginas)


@bp.get("")
@com_base_filtrada
def tela(df, _selecao):
    return render_template("dados.html", lista=paginar(df, COLUNAS_TABELA))
