from functools import wraps

from flask import flash, redirect, render_template, request, url_for

from app import estado, filtros

# a ordem da barra lateral
TELAS = [
    ("hipotese.tela", "A hipótese"),
    ("quando.tela", "Quando?"),
    ("onde.tela", "Onde?"),
    ("condicoes.tela", "Condições"),
    ("modelos.tela", "Modelos"),
    ("dados.tela", "Dados"),
]


def com_base_filtrada(view):
    """Entrega à tela a base com os filtros da barra lateral, ou desvia para a tela certa."""

    @wraps(view)
    def envolvida(*args, **kwargs):
        base = estado.base()
        if base is None:
            flash("Carregue uma base primeiro.", "aviso")
            return redirect(url_for("dados.tela"))

        selecao = filtros.ler(request.args)
        df = filtros.aplicar(base.limpo, selecao)
        if df.empty:
            return render_template("vazio.html")

        return view(df, selecao, *args, **kwargs)

    return envolvida
