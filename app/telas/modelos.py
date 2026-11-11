from flask import Blueprint, flash, redirect, render_template, url_for
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from markupsafe import Markup

from app import estado, filtros
from app.formato import num, pct
from app.telas import com_base_filtrada
from app.telas.dados import paginar
from colunas import ALVO
from modelos.projecao import projetar_2d

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
    rede = modelos.classificadores.get("Rede com embeddings (PyTorch)")
    if rede is None:
        return {"embeddings": None}

    # o teste justo é a mesma rede com one-hot; sem ele (treino antigo), a comparação cai para o MLP do scikit-learn
    onehot = modelos.classificadores.get("Rede com one-hot (PyTorch)")
    mesma_rede = onehot is not None
    comparado = onehot or modelos.classificadores.get("MLP (scikit-learn)")
    if comparado is None:
        return {"embeddings": None}

    fatias_rede = rede.metricas_fatias
    fatias_onehot = comparado.metricas_fatias
    recall_rede, recall_onehot = fatias_rede["recall"].mean(), fatias_onehot["recall"].mean()
    # a diferença só conta se for maior que a variação entre as fatias
    diferenca = (fatias_rede["recall"] - fatias_onehot["recall"])
    confirmou = diferenca.mean() > 0 and (diferenca > 0).all()
    nome_onehot = "a mesma rede com one-hot" if mesma_rede else "o MLP do scikit-learn com one-hot"
    return {
        "embeddings": {
            "mesma_rede": mesma_rede,
            "recall_onehot": recall_onehot * 100,
            "recall_rede": recall_rede * 100,
            "pr_auc_onehot": fatias_onehot["precisao_media"].mean(),
            "pr_auc_rede": fatias_rede["precisao_media"].mean(),
            "fatias_a_favor": int((diferenca > 0).sum()),
            "fatias": len(diferenca),
            "leitura": (
                f"A rede com embeddings encontra {pct(recall_rede * 100)} dos acidentes fatais, e {nome_onehot}, "
                f"{pct(recall_onehot * 100)}. Os embeddings ganham em {int((diferenca > 0).sum())} de {len(diferenca)} fatias: "
                f"a hipótese técnica {'se confirma' if confirmou else 'não se confirma'}."
            ),
        }
    }


COLUNA_PROJECAO = "causa_acidente"


def _contexto_projecao(rede, base: pd.DataFrame) -> dict:
    projecao, variancia = projetar_2d(rede.modelo.embeddings_[COLUNA_PROJECAO])

    # a cor de cada ponto: a % de acidentes fatais daquela causa na base
    letalidade = base.groupby(COLUNA_PROJECAO)[ALVO].mean() * 100
    projecao["letalidade"] = letalidade.reindex(projecao.index)

    fig = go.Figure(go.Scatter(
        x=projecao["x"], y=projecao["y"], mode="markers",
        text=projecao.index,
        marker=dict(size=9, color=projecao["letalidade"], colorscale="Reds", colorbar=dict(title="% fatais"),
                    line=dict(color="#52514e", width=0.5)),
        hovertemplate="%{text}<br>%{marker.color:.1f}% fatais<extra></extra>",
    ))
    fig.update_layout(template="simple_white", height=480, margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(family="system-ui, sans-serif", size=13))
    fig.update_xaxes(title="componente 1", showticklabels=False)
    fig.update_yaxes(title="componente 2", showticklabels=False)

    # só o pedaço da figura: a página e o plotly.js já vêm do base.html
    html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})
    return {"grafico_projecao": Markup(html), "variancia_projecao": variancia}


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
    rede = modelos.classificadores.get("Rede com embeddings (PyTorch)")
    if rede is not None:
        contexto.update(_contexto_projecao(rede, estado.base().limpo))
    contexto["tem_adicionados"] = not estado.dados_adicionados().empty
    if estado.previsao() is not None:
        contexto.update(_contexto_previsao(estado.previsao(), selecao))

    return render_template("modelos.html", **contexto)
