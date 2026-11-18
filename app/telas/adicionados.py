from xml.etree.ElementTree import ParseError

import pandas as pd
from flask import Blueprint, flash, redirect, render_template, request, url_for

from app import estado, filtros
from app.formato import num
from app.telas.dados import COLUNAS_TABELA, paginar
from validar import ErroDeColunas, ErroDeFormato

bp = Blueprint("adicionados", __name__, url_prefix="/adicionados")

# erros que vêm de um arquivo ruim; a mensagem vai para a tela e a base anterior continua
ERROS_DE_LEITURA = (ErroDeColunas, ErroDeFormato, ValueError, ParseError, pd.errors.ParserError, UnicodeDecodeError, KeyError)


@bp.get("")
def tela():
    adicionados = estado.dados_adicionados()
    lista = None
    if not adicionados.empty:
        lista = paginar(filtros.aplicar(adicionados, filtros.ler(request.args)), ["arquivo"] + COLUNAS_TABELA)

    return render_template("adicionados.html", lista=lista)


@bp.post("/upload")
def upload():
    arquivo = request.files.get("arquivo")
    if arquivo is None or not arquivo.filename:
        flash("Escolha um arquivo antes de enviar.", "erro")
        return redirect(url_for("adicionados.tela"))

    formato = arquivo.filename.rsplit(".", 1)[-1].lower() if "." in arquivo.filename else ""
    try:
        envio = estado.adicionar(arquivo.read(), arquivo.filename, formato)
    except ERROS_DE_LEITURA as erro:
        flash(f"O arquivo não foi adicionado. {erro}", "erro")
        return redirect(url_for("adicionados.tela"))

    repetidas = envio.linhas_arquivo - envio.linhas_novas
    if not envio.linhas_novas:
        flash(f"{envio.arquivo} não mudou a base: os {num(repetidas, 0)} acidentes do arquivo já estavam nela.", "aviso")
        return redirect(url_for("adicionados.tela"))

    flash(f"{envio.arquivo} adicionado à base: {num(envio.linhas_novas, 0)} acidentes novos"
          + (f" ({num(repetidas, 0)} já estavam na base)." if repetidas else "."), "sucesso")
    return redirect(url_for("adicionados.tela"))


@bp.post("/remover")
def remover():
    estado.remover_adicionados()
    flash("Os arquivos adicionados foram removidos. A base voltou a ser a base inicial.", "sucesso")
    return redirect(url_for("adicionados.tela"))
