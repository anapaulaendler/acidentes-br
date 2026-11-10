from flask import Blueprint, render_template

from analises.letalidade import letalidade_por
from app import graficos
from app.formato import pct
from app.telas import com_base_filtrada

bp = Blueprint("condicoes", __name__)

# categorias com poucos acidentes têm uma letalidade instável; ficam fora dos gráficos
MINIMO_CATEGORIA = 30


def _ordenada(tabela):
    tabela = tabela[tabela["acidentes"] >= MINIMO_CATEGORIA]
    return tabela["letalidade"].sort_values(ascending=False)


def _leitura_causas(por_causa) -> str:
    return (f"O grupo de causa com a maior letalidade é \"{por_causa.index[0]}\" ({pct(por_causa.iloc[0])}). "
            f"O grupo com a menor é \"{por_causa.index[-1]}\" ({pct(por_causa.iloc[-1])}).")


def _leitura_clima(por_clima) -> str:
    texto = (f"A condição com a maior letalidade é \"{por_clima.index[0]}\" ({pct(por_clima.iloc[0])}). "
             f"A condição com a menor é \"{por_clima.index[-1]}\" ({pct(por_clima.iloc[-1])}).")
    if {"Chuva", "Céu Claro"} <= set(por_clima.index):
        relacao = "mais baixa" if por_clima["Chuva"] < por_clima["Céu Claro"] else "mais alta"
        texto += (f" Com chuva, a letalidade é {relacao} que com céu claro ({pct(por_clima['Chuva'])} contra {pct(por_clima['Céu Claro'])})."
                  " Uma explicação possível: com chuva, os motoristas reduzem a velocidade.")
    return texto


@bp.get("/condicoes")
@com_base_filtrada
def tela(df, _selecao):
    por_clima = _ordenada(letalidade_por(df, "condicao_metereologica"))
    por_causa = _ordenada(letalidade_por(df, "grupo_causa"))
    tracado = df.assign(parte=df["tracado_via"].str.split(";")).explode("parte")
    por_tracado = _ordenada(letalidade_por(tracado, "parte"))

    contexto = {
        "grafico_clima": graficos.barras(por_clima, "% dos acidentes que são fatais"),
        "leitura_clima": _leitura_clima(por_clima) if len(por_clima) else None,
        "grafico_causa": graficos.barras(por_causa, "% dos acidentes que são fatais", horizontal=True, altura=380),
        "leitura_causas": _leitura_causas(por_causa) if len(por_causa) else None,
        "grafico_tracado": graficos.barras(por_tracado, "% dos acidentes que são fatais"),
        "minimo_categoria": MINIMO_CATEGORIA,
    }

    return render_template("condicoes.html", **contexto)
