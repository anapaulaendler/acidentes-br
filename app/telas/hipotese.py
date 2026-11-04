from flask import Blueprint, render_template

from analises.letalidade import frontal_por_pista, letalidade_por, letalidade_por_ano_e_pista
from app import graficos
from app.formato import num, pct
from app.telas import com_base_filtrada
from colunas import ALVO

bp = Blueprint("hipotese", __name__)

ORDEM_PISTA = ["Simples", "Dupla", "Múltipla"]


def _na_ordem(tabela):
    return tabela.reindex([p for p in ORDEM_PISTA if p in tabela.index])


def _leitura_letalidade(letalidade) -> str:
    if not {"Simples", "Dupla"} <= set(letalidade.index):
        return "Com os filtros atuais, a base não tem os dois tipos de pista (simples e dupla) para comparar."

    simples, dupla = letalidade["Simples"], letalidade["Dupla"]
    texto = f"Na pista simples, {pct(simples)} dos acidentes são fatais. Na pista dupla, {pct(dupla)}."
    if dupla > 0:
        texto += f" A letalidade da pista simples é {num(simples / dupla)} vezes a da pista dupla."
    return texto


def _leitura_frontal(frontal) -> str:
    if not {"Simples", "Dupla"} <= set(frontal.index):
        return "Com os filtros atuais, a base não tem os dois tipos de pista para comparar."

    s, d = frontal.loc["Simples"], frontal.loc["Dupla"]
    return (f"A colisão frontal é {pct(s['pct_frontal'])} dos acidentes na pista simples e {pct(d['pct_frontal'])} na pista dupla. "
            f"Quando acontece, a colisão frontal é fatal em {pct(s['letalidade_frontal'])} dos casos na pista simples. "
            f"Sem separação física entre os sentidos, este tipo de colisão é mais frequente e mais grave.")


def _leitura_anos(por_ano) -> str:
    if not {"Simples", "Dupla"} <= set(por_ano.columns):
        return "Com os filtros atuais, a base não tem os dois tipos de pista para comparar."

    comparaveis = por_ano[["Simples", "Dupla"]].dropna()
    maior = int((comparaveis["Simples"] > comparaveis["Dupla"]).sum())
    return f"A pista simples tem a letalidade mais alta em {maior} de {len(comparaveis)} anos."


@bp.get("/")
@com_base_filtrada
def tela(df, selecao):
    letalidade = _na_ordem(letalidade_por(df, "tipo_pista"))
    frontal = _na_ordem(frontal_por_pista(df))
    por_ano = letalidade_por_ano_e_pista(df)[[p for p in ORDEM_PISTA if p in df["tipo_pista"].unique()]]

    return render_template(
        "hipotese.html",
        total=len(df),
        pct_fatal=df[ALVO].mean() * 100,
        anos=f"{df['ano'].min()}–{df['ano'].max()}",
        grafico_letalidade=graficos.barras(letalidade["letalidade"], "% dos acidentes que são fatais", graficos.CORES_PISTA),
        leitura_letalidade=_leitura_letalidade(letalidade["letalidade"]),
        grafico_frontal_freq=graficos.barras(frontal["pct_frontal"], "% colisão frontal", graficos.CORES_PISTA, altura=280),
        grafico_frontal_letal=graficos.barras(frontal["letalidade_frontal"], "% fatais", graficos.CORES_PISTA, altura=280),
        leitura_frontal=_leitura_frontal(frontal),
        grafico_anos=graficos.linhas(por_ano, "% dos acidentes que são fatais", graficos.CORES_PISTA),
        leitura_anos=_leitura_anos(por_ano),
        # a frase sobre 2017 só aparece se, em cada pista, 2017 tem a letalidade mais baixa da série
        destaque_2017=bool(2017 in por_ano.index and len(por_ano) > 1 and (por_ano.loc[2017] <= por_ano.min()).all()),
    )
