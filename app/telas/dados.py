from xml.etree.ElementTree import ParseError

import pandas as pd
from flask import Blueprint, flash, redirect, render_template, request, url_for

from analises.limpeza_relatorio import causas_por_ano, proporcao_grupos_por_ano, relatorio_limpeza
from app import estado, filtros, graficos
from validar import ErroDeColunas, ErroDeFormato

bp = Blueprint("dados", __name__, url_prefix="/dados")

LINHAS_POR_PAGINA = 50

COLUNAS_TABELA = ["data_inversa", "horario", "uf", "br", "km", "municipio", "causa_acidente", "grupo_causa",
                  "tipo_acidente", "tipo_pista", "condicao_metereologica", "fatal"]

# erros que vêm de um arquivo ruim; a mensagem vai para a tela e a base anterior continua
ERROS_DE_LEITURA = (ErroDeColunas, ErroDeFormato, ValueError, ParseError, pd.errors.ParserError, UnicodeDecodeError, KeyError)


def _resultado_da_leitura() -> None:
    base = estado.base()
    removidas = len(base.bruto) - len(base.limpo)
    flash(f"Base carregada: {base.nome}. {len(base.limpo):,} acidentes depois da limpeza ({removidas} linhas repetidas removidas).".replace(",", "."), "sucesso")


@bp.get("")
def tela():
    base = estado.base()
    if base is None:
        return render_template("dados.html", tem_base_completa=estado.tem_base_completa())

    selecao = filtros.ler(request.args)
    df = filtros.aplicar(base.limpo, selecao)

    relatorio = estado.em_cache(("relatorio",), lambda: relatorio_limpeza(base.bruto, base.limpo))
    causas = estado.em_cache(("causas_por_ano",), lambda: causas_por_ano(base.limpo))
    grupos = estado.em_cache(("grupos_por_ano",), lambda: proporcao_grupos_por_ano(base.limpo))

    paginas = max(1, -(-len(df) // LINHAS_POR_PAGINA))
    pagina = min(max(request.args.get("pagina", 1, type=int), 1), paginas)
    inicio = (pagina - 1) * LINHAS_POR_PAGINA
    trecho_tabela = df[COLUNAS_TABELA].iloc[inicio:inicio + LINHAS_POR_PAGINA].copy()
    trecho_tabela["data_inversa"] = trecho_tabela["data_inversa"].dt.strftime("%d/%m/%Y")

    return render_template(
        "dados.html",
        tem_base_completa=estado.tem_base_completa(),
        relatorio=relatorio,
        grafico_causas=graficos.barras(causas.rename(index=str), "Causas diferentes", sufixo="", casas=0),
        maximo_causas=causas.max(),
        tem_quebra_2021={2020, 2021} <= set(causas.index),
        minimo_causas=causas.min(),
        grupos=grupos.rename_axis(index="grupo de causa", columns="ano"),
        linhas_filtradas=len(df),
        tabela=trecho_tabela,
        pagina=pagina,
        paginas=paginas,
    )


@bp.post("/upload")
def upload():
    arquivo = request.files.get("arquivo")
    if arquivo is None or not arquivo.filename:
        flash("Escolha um arquivo antes de enviar.", "erro")
        return redirect(url_for("dados.tela"))

    formato = arquivo.filename.rsplit(".", 1)[-1].lower() if "." in arquivo.filename else ""
    try:
        estado.carregar_base(arquivo.stream, arquivo.filename, formato)
    except ERROS_DE_LEITURA as erro:
        flash(f"O arquivo não foi carregado. {erro}", "erro")
        return redirect(url_for("dados.tela"))

    _resultado_da_leitura()
    return redirect(url_for("dados.tela"))


@bp.post("/amostra")
def amostra():
    estado.carregar_base(estado.AMOSTRA, "Amostra PR 2024")
    _resultado_da_leitura()
    return redirect(url_for("dados.tela"))


@bp.post("/completa")
def completa():
    estado.carregar_base_completa()
    _resultado_da_leitura()
    return redirect(url_for("dados.tela"))
