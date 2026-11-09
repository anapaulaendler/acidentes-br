from flask import Blueprint, render_template

from app.telas import com_base_filtrada
from colunas import ALVO

bp = Blueprint("onde", __name__)

# acima disso, o navegador fica lento; uma amostra aleatória mantém a distribuição no mapa
MAXIMO_PONTOS = 15_000


def _pontos_do_mapa(df) -> tuple[list, int, int]:
    fatais = df[df[ALVO]].dropna(subset=["latitude", "longitude"])
    total = len(fatais)
    if total > MAXIMO_PONTOS:
        fatais = fatais.sample(MAXIMO_PONTOS, random_state=42)

    pontos = [
        [round(lat, 5), round(lon, 5), f"{municipio} ({uf}), BR-{br} · {data:%d/%m/%Y} · {tipo}"]
        for lat, lon, municipio, uf, br, data, tipo in fatais[
            ["latitude", "longitude", "municipio", "uf", "br", "data_inversa", "tipo_acidente"]
        ].itertuples(index=False)
    ]
    return pontos, len(pontos), total


@bp.get("/onde")
@com_base_filtrada
def tela(df, _selecao):
    pontos, mostrados, total_fatais = _pontos_do_mapa(df)
    return render_template("onde.html", pontos=pontos, pontos_mostrados=mostrados, total_fatais=total_fatais)
