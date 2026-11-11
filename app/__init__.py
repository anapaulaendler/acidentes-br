from flask import Flask, request

from app import estado, filtros, formato

LIMITE_UPLOAD_MB = 300


def criar_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = "acidentes-prf-demonstracao-local"
    app.config["MAX_CONTENT_LENGTH"] = LIMITE_UPLOAD_MB * 1024 * 1024

    estado.iniciar()

    from app.telas import TELAS, acidentes, dados, modelos

    for modulo in (acidentes, modelos, dados):
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

    return app
