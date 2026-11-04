import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from markupsafe import Markup

AZUL, LARANJA, VERDE, AMARELO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CINZA_TEXTO, CINZA_GRADE = "#52514e", "#e5e4e0"

CORES_PISTA = {"Simples": LARANJA, "Dupla": AZUL, "Múltipla": VERDE}
CORES_SERIE = [AZUL, LARANJA, VERDE, AMARELO]


def _layout(fig: go.Figure, titulo_y: str = "", altura: int = 320) -> go.Figure:
    fig.update_layout(
        template="simple_white",
        height=altura,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="system-ui, sans-serif", size=13, color=CINZA_TEXTO),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_yaxes(title=titulo_y, gridcolor=CINZA_GRADE, showgrid=True)
    return fig


def html(fig: go.Figure) -> Markup:
    return Markup(pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True}))


def barras(serie: pd.Series, titulo_y: str = "", cores: dict[str, str] | None = None, sufixo: str = "%",
           horizontal: bool = False, altura: int = 320, casas: int = 1) -> Markup:
    rotulos = [str(r) for r in serie.index]
    cor = [cores.get(r, AZUL) for r in rotulos] if cores else AZUL
    texto = ["" if pd.isna(v) else f"{v:,.{casas}f}{sufixo}".replace(",", "_").replace(".", ",").replace("_", ".") for v in serie.values]

    if horizontal:
        barra = go.Bar(x=serie.values, y=rotulos, orientation="h", marker_color=cor, text=texto, textposition="outside",
                       hovertemplate="%{y}: %{text}<extra></extra>")
    else:
        barra = go.Bar(x=rotulos, y=serie.values, marker_color=cor, text=texto, textposition="outside",
                       hovertemplate="%{x}: %{text}<extra></extra>")

    fig = _layout(go.Figure(barra), titulo_y, altura)
    # espaço para o rótulo fora da barra não ser cortado na borda
    alcance = [min(0, serie.min() * 1.25), max(0, serie.max() * 1.18)]
    if horizontal:
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_xaxes(title=titulo_y, range=alcance)
        fig.update_yaxes(title="", showgrid=False)
    else:
        fig.update_yaxes(range=alcance)
    fig.update_traces(cliponaxis=False)
    return html(fig)


def barras_agrupadas(df: pd.DataFrame, titulo: str = "", cores: dict[str, str] | None = None, altura: int = 340) -> Markup:
    # barras horizontais: os nomes longos das categorias ficam legíveis no eixo vertical
    fig = go.Figure()
    for i, coluna in enumerate(df.columns):
        cor = (cores or {}).get(coluna, CORES_SERIE[i % len(CORES_SERIE)])
        texto = ["" if pd.isna(v) else f"{v:.1f}%".replace(".", ",") for v in df[coluna]]
        fig.add_bar(name=str(coluna), y=[str(r) for r in df.index], x=df[coluna], orientation="h", marker_color=cor,
                    text=texto, textposition="outside", cliponaxis=False,
                    hovertemplate=f"{coluna}<br>%{{y}}: %{{x:.1f}}%<extra></extra>")

    fig = _layout(fig, "", altura)
    fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"))
    maximo = df.max().max()
    fig.update_xaxes(title=titulo, range=[0, (0 if pd.isna(maximo) else float(maximo)) * 1.2 + 1])
    fig.update_yaxes(showgrid=False)
    return html(fig)


def linhas(df: pd.DataFrame, titulo_y: str = "", cores: dict[str, str] | None = None, altura: int = 320) -> Markup:
    fig = go.Figure()
    for i, coluna in enumerate(df.columns):
        cor = (cores or {}).get(coluna, CORES_SERIE[i % len(CORES_SERIE)])
        fig.add_scatter(name=str(coluna), x=df.index, y=df[coluna], mode="lines+markers", line=dict(color=cor, width=2),
                        marker=dict(size=8), hovertemplate=f"{coluna}<br>%{{x}}: %{{y:.1f}}<extra></extra>")

    fig.update_xaxes(dtick=1)
    return html(_layout(fig, titulo_y, altura))


def pizza(serie: pd.Series, altura: int = 320) -> Markup:
    fig = go.Figure(go.Pie(labels=[str(r) for r in serie.index], values=serie.values, hole=0.45, sort=False,
                           marker=dict(colors=CORES_SERIE, line=dict(color="white", width=2))))
    return html(_layout(fig, altura=altura))


def dispersao_previsto(real: pd.Series, previsto, altura: int = 360) -> Markup:
    limite = float(max(real.max(), previsto.max()))
    fig = go.Figure()
    fig.add_scatter(x=real, y=previsto, mode="markers", marker=dict(color=AZUL, size=6, opacity=0.4),
                    name="trechos do teste", hovertemplate="real %{x:.1f}%<br>prevista %{y:.1f}%<extra></extra>")
    fig.add_scatter(x=[0, limite], y=[0, limite], mode="lines", line=dict(color=CINZA_TEXTO, dash="dash", width=1),
                    name="previsão = real", hoverinfo="skip")
    fig = _layout(fig, "Letalidade prevista (%)", altura)
    fig.update_xaxes(title="Letalidade real (%)")
    return html(fig)


def dispersao_personas(trechos: pd.DataFrame, cores: dict[str, str], altura: int = 420) -> Markup:
    fig = go.Figure()
    for persona, grupo in trechos.groupby("persona"):
        fig.add_scatter(x=grupo["acidentes_por_ano"], y=grupo["pct_curva"], mode="markers", name=persona,
                        marker=dict(color=cores.get(persona, CINZA_TEXTO), size=6, opacity=0.5),
                        hovertemplate=f"{persona}<br>%{{x:.1f}} acidentes/ano<br>%{{y:.0f}}% com curva<extra></extra>")
    fig = _layout(fig, "% dos acidentes com curva", altura)
    fig.update_xaxes(title="Acidentes por ano (escala log)", type="log", tickvals=[5, 10, 20, 50, 100, 200])
    return html(fig)


def cotovelo_silhueta(diagnostico: pd.DataFrame, k_escolhido: int) -> tuple[Markup, Markup]:
    graficos = []
    for coluna, titulo in [("inércia", "Inércia (WCSS)"), ("silhueta", "Silhueta média")]:
        fig = go.Figure(go.Scatter(x=diagnostico.index, y=diagnostico[coluna], mode="lines+markers",
                                   line=dict(color=AZUL, width=2), marker=dict(size=8),
                                   hovertemplate="K = %{x}<br>%{y:.3f}<extra></extra>"))
        fig.add_vline(x=k_escolhido, line_dash="dash", line_color=CINZA_TEXTO)
        fig = _layout(fig, titulo, 280)
        fig.update_xaxes(title="K (número de grupos)", dtick=1)
        graficos.append(html(fig))
    return graficos[0], graficos[1]
