from flask import Blueprint, render_template

from analises.letalidade import letalidade_por
from app import calculos, graficos
from app.formato import num, pct
from app.telas import com_base_filtrada
from modelos import regressao_trechos

bp = Blueprint("condicoes", __name__)

# categorias com poucos acidentes têm uma letalidade instável; ficam fora dos gráficos
MINIMO_CATEGORIA = 30

NOMES_VARIAVEIS = {
    "pct_pista_simples": "% pista simples",
    "pct_curva": "% curva",
    "pct_noite": "% noite",
    "pct_chuva": "% chuva",
    "pct_urbano": "% urbano",
    "pct_velocidade": "% causa: velocidade",
    "pct_alcool": "% causa: álcool e drogas",
    "pct_pedestre": "% causa: pedestre",
    "acidentes_por_ano": "acidentes por ano",
    "regiao_Nordeste": "região: Nordeste",
    "regiao_Norte": "região: Norte",
    "regiao_Sudeste": "região: Sudeste",
    "regiao_Sul": "região: Sul",
}


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


SEM_EFEITO_DA_PISTA = ("Com o filtro de pista, todos os trechos têm o mesmo tipo de pista: a regressão não mede o efeito da pista. "
                       "Os outros coeficientes continuam válidos.")


def _contexto_regressao(resultado) -> dict:
    efeito = resultado.coeficientes.loc["pct_pista_simples", "efeito de 0% → 100%"]
    sensibilidade = resultado.sensibilidade["pista simples 0% → 100% (pp)"].dropna()
    teste = resultado.metricas.loc["teste"]
    base = resultado.metricas.loc["base (média) no teste"]

    efeitos = resultado.coeficientes["efeito de 0% → 100%"].dropna().rename(index=NOMES_VARIAVEIS).sort_values(ascending=False)

    return {
        "regressao_trechos": resultado.trechos,
        "grafico_coeficientes": graficos.barras(efeitos, "Efeito na letalidade (pp)", sufixo=" pp", horizontal=True, altura=360),
        "coeficientes": resultado.coeficientes.rename(index=NOMES_VARIAVEIS),
        "grafico_previsto": graficos.dispersao_previsto(resultado.real, resultado.previsto),
        "metricas": resultado.metricas,
        "sensibilidade": resultado.sensibilidade.set_index(["mínimo", "peso"]),
        "leitura_regressao": SEM_EFEITO_DA_PISTA if efeito != efeito else (
            f"Com os outros fatores iguais, um trecho só com pista simples tem {num(efeito)} pp a mais de letalidade "
            f"que um trecho sem pista simples."),
        "leitura_metricas": (f"No teste, o erro médio (MAE) é {num(teste['MAE'], 2)} pp, contra {num(base['MAE'], 2)} pp de um modelo que sempre "
                             f"prevê a média. O R² é {num(teste['R²'], 2)}: o modelo explica parte da variação, mas muitos fatores não estão na base."),
        "leitura_sensibilidade": SEM_EFEITO_DA_PISTA if sensibilidade.empty else (
            ("O efeito da pista simples é positivo em todas as combinações. " if (sensibilidade > 0).all()
             else "O efeito da pista simples muda de sinal em algumas combinações. ")
            + f"O valor varia de {num(sensibilidade.min())} a {num(sensibilidade.max())} pp, conforme o mínimo de acidentes e o peso. "
            "Por isso, o efeito é um intervalo, não um único número."),
    }


@bp.get("/condicoes")
@com_base_filtrada
def tela(df, selecao):
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
        "trechos_validos": calculos.trechos_validos(df, selecao),
        "minimo_trechos": regressao_trechos.MINIMO_TRECHOS,
    }

    resultado = calculos.regressao(df, selecao)
    if resultado is not None:
        contexto.update(_contexto_regressao(resultado))
    contexto["tem_regressao"] = resultado is not None

    return render_template("condicoes.html", **contexto)
