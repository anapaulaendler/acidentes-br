from functools import cache

from flask import Flask, Response, request
from plotly.offline import get_plotlyjs

from app import estado, filtros, formato

LIMITE_UPLOAD_MB = 300


@cache
def _plotly_js() -> str:
    return get_plotlyjs()


def criar_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = "acidentes-prf-demonstracao-local"
    app.config["MAX_CONTENT_LENGTH"] = LIMITE_UPLOAD_MB * 1024 * 1024

    from app.telas import TELAS, condicoes, dados, hipotese, modelos, onde, quando

    for modulo in (hipotese, quando, onde, condicoes, modelos, dados):
        app.register_blueprint(modulo.bp)

    app.add_template_filter(formato.num, "num")
    app.add_template_filter(formato.pct, "pct")
    app.add_template_filter(formato.tabela, "tabela")

    @app.context_processor
    def barra_lateral():
        base = estado.base()
        return {
            "telas": TELAS,
            "base": base,
            "filtros_atuais": filtros.ler(request.args),
            "opcoes_filtro": filtros.opcoes(base.limpo) if base else None,
        }

    @app.get("/plotly.js")
    def plotly_js():
        return Response(_plotly_js(), mimetype="application/javascript", headers={"Cache-Control": "max-age=86400"})

    return app
