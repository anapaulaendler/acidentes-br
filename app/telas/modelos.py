from flask import Blueprint, render_template
import pandas as pd

from analises.comparacao import NOMES_METRICAS
from app import estado
from app.formato import num, pct
from app.telas import com_base_filtrada

bp = Blueprint("modelos", __name__)


def _contexto_classificadores(modelos) -> dict:
    linhas = {}
    for nome, classificador in modelos.classificadores.items():
        fatias = classificador.metricas_fatias.rename(columns=NOMES_METRICAS)
        linhas[nome] = {
            metrica: f"{num(media, 3)} ± {num(desvio, 3)}"
            for metrica, media, desvio in zip(fatias.columns, fatias.mean(), fatias.std())
        }

    recall = {nome: c.metricas_fatias["recall"].mean() for nome, c in modelos.classificadores.items()}
    melhor = max(recall, key=recall.get)
    return {
        "fatias": len(next(iter(modelos.classificadores.values())).metricas_fatias),
        "tabela_classificadores": pd.DataFrame.from_dict(linhas, orient="index"),
        "leitura_classificadores": (
            f"O modelo que encontra mais acidentes fatais é {melhor} ({pct(recall[melhor] * 100)}). "
            "Com uma classe rara, a acurácia engana: o recall e a precisão mostram o desempenho real."
        ),
    }


@bp.get("/modelos")
@com_base_filtrada
def tela(df, selecao):
    modelos = estado.modelos()
    if modelos is None:
        return render_template("modelos.html", modelos=None)

    contexto = {"modelos": modelos, "base_atual": len(estado.base().limpo)}
    contexto.update(_contexto_classificadores(modelos))

    return render_template("modelos.html", **contexto)
