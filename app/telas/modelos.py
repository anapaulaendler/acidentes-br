from flask import Blueprint, render_template

from app import calculos, graficos
from app.formato import num, pct
from app.telas import com_base_filtrada
from modelos import personas, regressao_trechos
from modelos.classificacao import MINIMO_FATAIS

bp = Blueprint("modelos", __name__)

NOMES_METRICAS = {
    "recall": "recall",
    "precisao": "precisão",
    "f1": "F1",
    "precisao_media": "PR-AUC",
    "acuracia": "acurácia",
}


def _contexto_regressao(df, selecao) -> dict:
    resultado = calculos.regressao(df, selecao)
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


def _contexto_kmeans(df, selecao) -> dict:
    diagnostico = calculos.diagnostico_k(df, selecao)
    agrupados = calculos.agrupamento(df, selecao)
    if diagnostico is None or agrupados is None:
        return {"kmeans": None}

    comparacao = personas.comparar_pistas(agrupados).dropna(subset=["diferença (pp)"])
    grafico_inercia, grafico_silhueta = graficos.cotovelo_silhueta(diagnostico, personas.K)
    melhor_silhueta = int(diagnostico["silhueta"].idxmax())
    return {
        "kmeans": True,
        "k_maior_silhueta": melhor_silhueta,
        "grafico_inercia": grafico_inercia,
        "grafico_silhueta": grafico_silhueta,
        "silhueta_k": num(diagnostico.loc[personas.K, "silhueta"], 3),
        "trechos_kmeans": len(agrupados),
        "diferencas": comparacao["diferença (pp)"],
    }


def _contexto_classificacao(df, selecao) -> dict:
    resultado = calculos.classificacao(df, selecao)
    if resultado is None:
        return {"metricas_classificacao": None, "fatais_base": int(df["fatal"].sum())}

    metricas = resultado.metricas.rename(columns=NOMES_METRICAS)

    logistica = metricas.loc["Regressão logística"]
    mlp = metricas.loc["MLP (scikit-learn)"]
    return {
        "metricas_classificacao": metricas,
        "linhas_classificacao": resultado.linhas_usadas,
        "amostrou": resultado.linhas_usadas < len(df),
        "odds": num(resultado.odds_pista_simples, 2) if resultado.odds_pista_simples else None,
        "leitura_classificacao": (
            f"A regressão logística (com pesos de classe) encontra {pct(logistica['recall'] * 100)} dos acidentes fatais, "
            f"mas só {pct(logistica['precisão'] * 100)} dos alertas dela são fatais de verdade. "
            f"O MLP (sem pesos de classe) tem acurácia de {pct(mlp['acurácia'] * 100)}, mas encontra só {pct(mlp['recall'] * 100)} dos fatais. "
            "Com uma classe rara, a acurácia engana: o recall e a precisão mostram o desempenho real."
        ),
    }


@bp.get("/modelos")
@com_base_filtrada
def tela(df, selecao):
    contexto = {
        "trechos_validos": calculos.trechos_validos(df, selecao),
        "minimo_regressao": regressao_trechos.MINIMO_TRECHOS,
        "minimo_kmeans": personas.MINIMO_TRECHOS,
        "minimo_fatais": MINIMO_FATAIS,
    }
    contexto.update(_contexto_regressao(df, selecao))
    contexto.update(_contexto_kmeans(df, selecao))
    contexto.update(_contexto_classificacao(df, selecao))

    return render_template("modelos.html", **contexto)
