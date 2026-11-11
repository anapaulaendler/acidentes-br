from math import ceil

from flask import Blueprint, render_template, request

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


@bp.get("/")
@bp.get("/acidentes")
@com_base_filtrada
def tela(df, _selecao):
    pontos, total, pagina, paginas = _pontos_do_mapa(df, request.args.get("pagina", 1, type=int))
    inicio = (pagina - 1) * POR_PAGINA
    return render_template("acidentes.html", pontos=pontos, total=total, pagina=pagina, paginas=paginas,
                           primeiro=inicio + 1, ultimo=inicio + len(pontos))
