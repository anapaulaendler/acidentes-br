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
