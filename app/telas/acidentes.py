from math import ceil

from flask import Blueprint, render_template, request

from app.formato import num, pct
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


@bp.get("/")
@bp.get("/acidentes")
@com_base_filtrada
def tela(df, _selecao):
    pontos, total, pagina, paginas = _pontos_do_mapa(df, request.args.get("pagina", 1, type=int))
    inicio = (pagina - 1) * POR_PAGINA
    barras_uf, leitura_uf = _acidentes_por_uf(df)
    return render_template("acidentes.html", pontos=pontos, total=total, pagina=pagina, paginas=paginas,
                           primeiro=inicio + 1, ultimo=inicio + len(pontos),
                           barras_uf=barras_uf, leitura_uf=leitura_uf)
