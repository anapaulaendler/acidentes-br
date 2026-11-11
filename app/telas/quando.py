from flask import Blueprint, render_template

from analises.letalidade import letalidade_por
from app import graficos
from app.formato import pct
from app.telas import com_base_filtrada

bp = Blueprint("quando", __name__)

DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
GRAVIDADE = ["Sem Vítimas", "Com Vítimas Feridas", "Com Vítimas Fatais"]


@bp.get("/")
@bp.get("/quando")
@com_base_filtrada
def tela(df, selecao):
    df = df.assign(hora=df["horario"].str[:2].astype(int), mes=df["data_inversa"].dt.month)

    por_hora = letalidade_por(df, "hora").reindex(range(24))
    por_dia = letalidade_por(df, "dia_semana").reindex(DIAS)
    por_mes = letalidade_por(df, "mes").reindex(range(1, 13))
    por_mes.index = MESES
    gravidade = df["classificacao_acidente"].value_counts().reindex(GRAVIDADE).dropna()

    hora_mais_acidentes = int(por_hora["acidentes"].idxmax())
    hora_mais_letal = int(por_hora["letalidade"].idxmax())
    dia_mais_letal = por_dia["letalidade"].idxmax()

    return render_template(
        "quando.html",
        grafico_hora=graficos.barras(por_hora["letalidade"].rename(index=lambda h: f"{h}h"), "% dos acidentes que são fatais"),
        grafico_hora_volume=graficos.barras(por_hora["acidentes"].rename(index=lambda h: f"{h}h"), "Acidentes", sufixo="", casas=0),
        leitura_hora=(f"O maior número de acidentes acontece às {hora_mais_acidentes}h. A maior letalidade acontece às {hora_mais_letal}h "
                      f"({pct(por_hora['letalidade'].max())} dos acidentes são fatais)."
                      + (" Os acidentes da madrugada são menos numerosos, mas mais graves." if hora_mais_letal < 6 else "")),
        grafico_dia=graficos.barras(por_dia["letalidade"], "% dos acidentes que são fatais"),
        leitura_dia=f"O dia com a maior letalidade é {dia_mais_letal} ({pct(por_dia['letalidade'].max())}).",
        grafico_mes=graficos.barras(por_mes["letalidade"], "% dos acidentes que são fatais"),
        grafico_gravidade=graficos.pizza(gravidade),
        leitura_gravidade=(f"{pct(gravidade.get('Com Vítimas Fatais', 0) / gravidade.sum() * 100)} dos acidentes têm pelo menos uma morte. "
                           f"Esta é a classe que os modelos tentam prever: uma classe rara."),
    )
