from flask import Blueprint, render_template

from analises.comparacao import NOMES_METRICAS
from app import estado
from app.formato import num, pct
from app.telas import com_base_filtrada

bp = Blueprint("modelos", __name__)


def _contexto_regressao(resultado) -> dict:
    if resultado is None:
        return {"regressao": None}

    teste = resultado.metricas.loc["teste"]
    sensibilidade = resultado.sensibilidade["pista simples 0% → 100% (pp)"].dropna()
    efeito = resultado.coeficientes.loc["pct_pista_simples", "efeito de 0% → 100%"]
    return {
        "regressao": resultado,
        "variaveis_regressao": len(resultado.coeficientes),
        "efeito_simples": None if efeito != efeito else f"{'+' if efeito >= 0 else ''}{num(efeito)}",
        "intervalo_simples": f"{num(sensibilidade.min())} a {num(sensibilidade.max())} pp" if len(sensibilidade) else None,
        "r2_regressao": num(teste["R²"], 2),
        "mae_regressao": num(teste["MAE"], 2),
    }


def _contexto_classificador(modelos) -> dict:
    metricas = modelos.metricas_classificador.rename(NOMES_METRICAS)
    return {
        "metricas_classificador": metricas.to_frame("teste"),
        "leitura_classificador": (
            f"O MLP tem acurácia de {pct(metricas['acurácia'] * 100)}, mas encontra só {pct(metricas['recall'] * 100)} dos acidentes fatais. "
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
    contexto.update(_contexto_regressao(modelos.regressao))
    contexto.update(_contexto_classificador(modelos))

    return render_template("modelos.html", **contexto)
