from flask import Blueprint, flash, redirect, render_template, url_for
import pandas as pd

from app import estado, filtros
from app.formato import num, pct
from app.telas import com_base_filtrada
from app.telas.dados import paginar
from colunas import ALVO

bp = Blueprint("modelos", __name__)

NOMES_METRICAS = {
    "recall": "recall",
    "precisao": "precisão",
    "f1": "F1",
    "precisao_media": "PR-AUC",
    "acuracia": "acurácia",
}


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
        ),
    }


def _contexto_embeddings(modelos) -> dict:
    mlp = modelos.classificadores.get("MLP (scikit-learn)")
    rede = modelos.classificadores.get("Rede com embeddings (PyTorch)")
    if mlp is None or rede is None:
        return {"embeddings": None}

    recall_mlp = mlp.metricas_fatias["recall"].mean()
    recall_rede = rede.metricas_fatias["recall"].mean()
    confirmou = recall_rede > recall_mlp
    return {
        "embeddings": {
            "recall_mlp": recall_mlp * 100,
            "recall_rede": recall_rede * 100,
            "leitura": (
                f"A rede com embeddings encontra {pct(recall_rede * 100)} dos acidentes fatais, e o MLP com one-hot, "
                f"{pct(recall_mlp * 100)}. A hipótese técnica {'se confirma' if confirmou else 'não se confirma'}."
            ),
        }
    }


# as colunas do acidente na lista de previsões; depois delas, uma coluna por modelo
COLUNAS_ACIDENTE = ["data_inversa", "uf", "br", "municipio", "tipo_acidente", "fatal"]


def _contexto_previsao(previsao, selecao) -> dict:
    tabela = previsao.metricas.rename(columns=NOMES_METRICAS).map(lambda v: num(v, 3))

    dados = previsao.previstos
    tem_fatais, so_fatais = dados[ALVO].any(), dados[ALVO].all()
    if not tem_fatais or so_fatais:
        leitura = (f"Os dados adicionados {'não têm acidentes fatais' if not tem_fatais else 'só têm acidentes fatais'}: "
                   "o recall, a precisão e a PR-AUC não dizem nada aqui. Só a acurácia vale.")
    else:
        recall = previsao.metricas["recall"]
        melhor = recall.idxmax()
        leitura = (f"Nos {num(len(dados), 0)} acidentes adicionados, o modelo que encontra mais acidentes fatais é {melhor}: "
                   f"{pct(recall[melhor] * 100)} dos {num(dados[ALVO].sum(), 0)} fatais. "
                   "Compare com a tabela de cima: se o recall cai muito, os dados novos fogem do padrão do treino.")

    # cada modelo vira uma coluna de texto: o que ele previu, com ✓ (acertou) ou ✗ (errou)
    lista = filtros.aplicar(dados, selecao)
    nomes = list(previsao.metricas.index)
    lista = lista.assign(**{
        nome: [("✓ " if previsto == real else "✗ ") + ("fatal" if previsto else "não fatal")
               for previsto, real in zip(lista[nome], lista[ALVO])]
        for nome in nomes
    })

    return {
        "previsao": previsao,
        "tabela_previsao": tabela,
        "leitura_previsao": leitura,
        "lista_previsao": paginar(lista, COLUNAS_ACIDENTE + nomes),
    }


@bp.post("/modelos/prever")
def prever():
    if estado.prever() is None:
        flash("Não há dados adicionados para prever. Envie um arquivo na tela Dados adicionados.", "aviso")
    return redirect(url_for("modelos.tela") + "#previsao")


@bp.get("/modelos")
@com_base_filtrada
def tela(df, selecao):
    modelos = estado.modelos()
    if modelos is None:
        return render_template("modelos.html", modelos=None)

    contexto = {"modelos": modelos}
    contexto.update(_contexto_classificadores(modelos))
    contexto.update(_contexto_embeddings(modelos))
    contexto["tem_adicionados"] = not estado.dados_adicionados().empty
    if estado.previsao() is not None:
        contexto.update(_contexto_previsao(estado.previsao(), selecao))

    return render_template("modelos.html", **contexto)
