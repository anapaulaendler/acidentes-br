from math import ceil

from flask import Blueprint, render_template, request
import plotly.graph_objects as go

from app.formato import grafico, num, pct
from app.telas import com_base_filtrada
from colunas import ALVO

bp = Blueprint("acidentes", __name__)

# acima disso, o navegador fica lento
POR_PAGINA = 15_000


def _pontos_do_mapa(df, pagina: int) -> tuple[list, int, int, int]:
    acidentes = df.dropna(subset=["latitude", "longitude"]).sort_values("data_inversa")
    total = len(acidentes)
    paginas = max(1, ceil(total / POR_PAGINA))
    pagina = min(max(1, pagina), paginas)

    inicio = (pagina - 1) * POR_PAGINA
    acidentes = acidentes.iloc[inicio:inicio + POR_PAGINA]

    pontos = [
        [round(lat, 5), round(lon, 5), bool(fatal),
         f"{municipio} ({uf}), BR-{br} · {data:%d/%m/%Y} · {tipo} · {'fatal' if fatal else 'não fatal'}"]
        for lat, lon, fatal, municipio, uf, br, data, tipo in acidentes[
            ["latitude", "longitude", ALVO, "municipio", "uf", "br", "data_inversa", "tipo_acidente"]
        ].itertuples(index=False)
    ]
    return pontos, total, pagina, paginas


def _acidentes_por_uf(df) -> tuple[list[dict], str]:
    """Quantos acidentes cada UF tem, separados em fatais e não fatais, do maior para o menor."""
    tabela = df.groupby("uf")[ALVO].agg(total="size", fatais="sum").sort_values("total", ascending=False)
    maior = tabela["total"].max()

    barras = [
        {"uf": uf, "total": linha.total, "fatais": linha.fatais, "nao_fatais": linha.total - linha.fatais,
         "pct_fatais": linha.fatais / linha.total * 100,
         # larguras em % da barra mais longa: a parte fatal e a não fatal, uma ao lado da outra
         "largura_fatais": linha.fatais / maior * 100, "largura_nao_fatais": (linha.total - linha.fatais) / maior * 100}
        for uf, linha in tabela.iterrows()
    ]

    mais = tabela.index[0]
    leitura = f"{mais} tem o maior número de acidentes ({num(tabela.loc[mais, 'total'], 0)})."
    if len(tabela) > 1 and tabela["fatais"].sum() and tabela["fatais"].sum() < tabela["total"].sum():
        letalidade = tabela["fatais"] / tabela["total"] * 100
        mais_letal = letalidade.idxmax()
        leitura += (f" A maior proporção de acidentes fatais é a de {mais_letal} ({pct(letalidade[mais_letal])}): "
                    "ter mais acidentes não quer dizer ter acidentes mais graves.")
    return barras, leitura


def _fatais_por_fase_do_dia(df) -> dict:
    """Pizza: em que fase do dia acontecem os acidentes fatais, comparado com todos os acidentes."""
    fatais = df.loc[df[ALVO], "fase_dia"].value_counts()
    if fatais.empty:
        return {"grafico_fase": None}

    fig = go.Figure(go.Pie(labels=fatais.index, values=fatais.values, hole=0.4, sort=False,
                           hovertemplate="%{label}: %{value} acidentes fatais (%{percent})<extra></extra>"))
    todos = df["fase_dia"].value_counts(normalize=True) * 100
    noite = "Plena Noite"
    leitura = None
    if noite in fatais.index:
        nos_fatais, em_todos = fatais[noite] / fatais.sum() * 100, todos.get(noite, 0)
        leitura = (f"{pct(nos_fatais)} dos acidentes fatais acontecem em plena noite, contra {pct(em_todos)} de todos os acidentes"
                   + (": a noite pesa mais nos fatais." if nos_fatais > em_todos else "."))
    return {"grafico_fase": grafico(fig), "leitura_fase": leitura}


def _letalidade_por_tipo(df) -> dict:
    """Barras: a % de acidentes fatais em cada tipo de acidente (agrupado)."""
    tabela = df.groupby("grupo_tipo")[ALVO].agg(total="size", pct_fatais="mean")
    tabela = tabela[tabela["total"] >= 30]  # ponytail: menos de 30 acidentes, a % não diz nada
    if tabela.empty or not df[ALVO].any():  # com o filtro "não fatais", tudo daria 0%
        return {"grafico_tipo": None}

    tabela["pct_fatais"] *= 100
    tabela = tabela.sort_values("pct_fatais")
    fig = go.Figure(go.Bar(x=tabela["pct_fatais"], y=tabela.index, orientation="h", marker_color="#eb6834",
                           customdata=tabela["total"],
                           hovertemplate="%{y}: %{x:.1f}% fatais em %{customdata} acidentes<extra></extra>"))
    fig.update_xaxes(title="% de acidentes fatais")
    mais = tabela.index[-1]
    leitura = f"{mais} é o tipo mais letal: {pct(tabela.loc[mais, 'pct_fatais'])} dos acidentes são fatais."
    return {"grafico_tipo": grafico(fig, altura=max(300, 28 * len(tabela))), "leitura_tipo": leitura}


@bp.get("/")
@bp.get("/acidentes")
@com_base_filtrada
def tela(df, _selecao):
    pontos, total, pagina, paginas = _pontos_do_mapa(df, request.args.get("pagina", 1, type=int))
    inicio = (pagina - 1) * POR_PAGINA
    barras_uf, leitura_uf = _acidentes_por_uf(df)
    return render_template("acidentes.html", pontos=pontos, total=total, pagina=pagina, paginas=paginas,
                           primeiro=inicio + 1, ultimo=inicio + len(pontos),
                           barras_uf=barras_uf, leitura_uf=leitura_uf,
                           **_fatais_por_fase_do_dia(df), **_letalidade_por_tipo(df))
