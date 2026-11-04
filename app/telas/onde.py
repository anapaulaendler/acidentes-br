from flask import Blueprint, render_template

from app import calculos, graficos
from app.formato import num, pct
from app.telas import com_base_filtrada
from colunas import ALVO
from modelos import personas

bp = Blueprint("onde", __name__)

# acima disso, o navegador fica lento; uma amostra aleatória mantém a distribuição no mapa
MAXIMO_PONTOS = 15_000

CORES_PERSONA = {
    "Corredores urbanos movimentados": graficos.AZUL,
    "Serra com curva e chuva": graficos.LARANJA,
    "Rurais com acidentes de vários veículos": graficos.VERDE,
    "Noite e fim de semana": graficos.AMARELO,
}

NOMES_PERFIL = {
    "trechos": "trechos",
    "acidentes_por_ano": "acidentes/ano",
    "pct_noite": "% noite",
    "pct_chuva": "% chuva",
    "pct_curva": "% curva",
    "pct_fim_de_semana": "% fim de semana",
    "media_veiculos": "veículos/acidente",
    "media_pessoas": "pessoas/acidente",
    "pct_urbano": "% urbano",
    "pct_pista_simples": "% pista simples",
    "letalidade_media": "letalidade média (%)",
}


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


def _leitura_comparacao(comparacao) -> str:
    validas = comparacao.dropna(subset=["diferença (pp)"])
    if validas.empty:
        return "Nenhuma persona tem trechos de pista simples e de pista dupla para comparar."

    maior, menor = validas["diferença (pp)"].idxmax(), validas["diferença (pp)"].idxmin()
    return (f"A maior diferença está em \"{maior}\": a pista simples tem {num(validas.loc[maior, 'diferença (pp)'])} pp a mais de letalidade. "
            f"A menor está em \"{menor}\" ({num(validas.loc[menor, 'diferença (pp)'])} pp). "
            f"Onde a diferença é pequena, a duplicação tem pouco efeito sobre a letalidade.")


@bp.get("/onde")
@com_base_filtrada
def tela(df, selecao):
    pontos, mostrados, total_fatais = _pontos_do_mapa(df)
    agrupados = calculos.agrupamento(df, selecao)

    contexto = {
        "pontos": pontos,
        "pontos_mostrados": mostrados,
        "total_fatais": total_fatais,
        "agrupados": agrupados is not None,
        "trechos_validos": calculos.trechos_validos(df, selecao),
        "minimo_trechos": personas.MINIMO_TRECHOS,
    }

    if agrupados is not None:
        perfil = personas.perfil(agrupados)
        comparacao = personas.comparar_pistas(agrupados)
        cores = {p: CORES_PERSONA.get(p, graficos.CINZA_TEXTO) for p in perfil.index}

        diagnostico = calculos.diagnostico_k(df, selecao)
        contexto.update(
            silhueta=num(diagnostico.loc[personas.K, "silhueta"], 3),
            mostra_separacao={"Corredores urbanos movimentados", "Serra com curva e chuva"} <= set(perfil.index),
            perfil=perfil.drop(columns="ação sugerida").rename(columns=NOMES_PERFIL),
            acoes=perfil["ação sugerida"],
            persona_mais_letal=perfil.index[0],
            letalidade_mais_letal=pct(perfil["letalidade_media"].iloc[0]),
            grafico_personas=graficos.dispersao_personas(agrupados, cores),
            grafico_comparacao=graficos.barras_agrupadas(
                comparacao[["simples", "dupla ou múltipla"]],
                "% dos acidentes que são fatais",
                {"simples": graficos.LARANJA, "dupla ou múltipla": graficos.AZUL},
            ),
            tabela_comparacao=comparacao,
            leitura_comparacao=_leitura_comparacao(comparacao),
        )

    return render_template("onde.html", **contexto)
