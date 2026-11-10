from flask import Blueprint, render_template

from analises.comparacao import NOMES_METRICAS
from app import estado
from app.formato import pct
from app.telas import com_base_filtrada

bp = Blueprint("modelos", __name__)


def _contexto_classificador(modelos) -> dict:
    fatias = modelos.metricas_fatias.rename(columns=NOMES_METRICAS)
    metricas = fatias.mean()
    tabela = fatias.T.rename(columns=lambda i: f"fatia {i + 1}")
    tabela["média"] = metricas
    tabela["desvio"] = fatias.std()
    return {
        "fatias": len(fatias),
        "metricas_classificador": tabela,
        "leitura_classificador": (
            f"Na média das {len(fatias)} fatias, o MLP tem acurácia de {pct(metricas['acurácia'] * 100)}, mas encontra só {pct(metricas['recall'] * 100)} dos acidentes fatais. "
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
    contexto.update(_contexto_classificador(modelos))

    return render_template("modelos.html", **contexto)
