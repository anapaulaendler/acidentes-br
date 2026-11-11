from math import ceil

from flask import Blueprint, render_template, request

from app.telas import com_base_filtrada
from colunas import ALVO

bp = Blueprint("onde", __name__)

# acima disso, o navegador fica lento
POR_PAGINA = 15_000


def _pontos_do_mapa(df, pagina: int) -> tuple[list, int, int, int]:
    fatais = df[df[ALVO]].dropna(subset=["latitude", "longitude"]).sort_values("data_inversa")
    total = len(fatais)
    paginas = max(1, ceil(total / POR_PAGINA))
    pagina = min(max(1, pagina), paginas)

    inicio = (pagina - 1) * POR_PAGINA
    fatais = fatais.iloc[inicio:inicio + POR_PAGINA]

    pontos = [
        [round(lat, 5), round(lon, 5), f"{municipio} ({uf}), BR-{br} · {data:%d/%m/%Y} · {tipo}"]
        for lat, lon, municipio, uf, br, data, tipo in fatais[
            ["latitude", "longitude", "municipio", "uf", "br", "data_inversa", "tipo_acidente"]
        ].itertuples(index=False)
    ]
    return pontos, total, pagina, paginas


@bp.get("/onde")
@com_base_filtrada
def tela(df, _selecao):
    pontos, total, pagina, paginas = _pontos_do_mapa(df, request.args.get("pagina", 1, type=int))
    inicio = (pagina - 1) * POR_PAGINA
    return render_template("onde.html", pontos=pontos, total_fatais=total, pagina=pagina, paginas=paginas,
                           primeiro=inicio + 1, ultimo=inicio + len(pontos))
