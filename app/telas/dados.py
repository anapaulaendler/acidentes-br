from xml.etree.ElementTree import ParseError

import pandas as pd
from flask import Blueprint, flash, redirect, render_template, request, url_for

from analises.limpeza_relatorio import causas_por_ano, proporcao_grupos_por_ano, relatorio_limpeza
from app import estado, filtros, graficos
from app.formato import num, pct
from validar import ErroDeColunas, ErroDeFormato

bp = Blueprint("dados", __name__, url_prefix="/dados")

LINHAS_POR_PAGINA = 50

COLUNAS_TABELA = ["data_inversa", "horario", "uf", "br", "km", "municipio", "causa_acidente", "grupo_causa",
                  "tipo_acidente", "tipo_pista", "condicao_metereologica", "fatal"]

# erros que vêm de um arquivo ruim; a mensagem vai para a tela e a base anterior continua
ERROS_DE_LEITURA = (ErroDeColunas, ErroDeFormato, ValueError, ParseError, pd.errors.ParserError, UnicodeDecodeError, KeyError)


# a diferença (em pontos percentuais) a partir da qual o lote novo foge do padrão da base
DIFERENCA_RELEVANTE_PP = 2
DIFERENCA_RELEVANTE_RECALL = 0.10


def _perfil_formatado(perfil):
    return perfil.map(lambda v: (pct(v) if v == v else "–") if isinstance(v, float) else v)


def _leitura_comparacao(comparacao) -> str:
    base, novos = comparacao.perfil["base antes do envio"], comparacao.perfil["dados novos"]
    diferenca = novos["% fatais"] - base["% fatais"]
    if abs(diferenca) < DIFERENCA_RELEVANTE_PP:
        texto = f"Os dados novos têm uma letalidade parecida com a da base ({pct(novos['% fatais'])} contra {pct(base['% fatais'])})."
    else:
        texto = (f"Os dados novos têm uma letalidade {'mais alta' if diferenca > 0 else 'mais baixa'} que a da base "
                 f"({pct(novos['% fatais'])} contra {pct(base['% fatais'])}).")

    if comparacao.modelo is not None:
        recall_validacao, recall_novos = comparacao.modelo.loc["recall"]
        if abs(recall_novos - recall_validacao) < DIFERENCA_RELEVANTE_RECALL:
            texto += (f" O modelo treinado encontra {pct(recall_novos * 100)} dos fatais dos dados novos, perto dos {pct(recall_validacao * 100)} da validação cruzada: "
                      "os dados novos seguem o padrão dos dados de treino.")
        else:
            texto += (f" O modelo treinado encontra {pct(recall_novos * 100)} dos fatais dos dados novos, contra {pct(recall_validacao * 100)} na validação cruzada: "
                      "os dados novos fogem do padrão dos dados de treino, e um novo treino pode mudar o modelo.")
    return texto


@bp.get("")
def tela():
    base = estado.base()
    comparacao = estado.ultima_comparacao()
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
        comparacao=comparacao,
        perfil_comparacao=_perfil_formatado(comparacao.perfil) if comparacao else None,
        leitura_comparacao=_leitura_comparacao(comparacao) if comparacao else None,
        modelos=estado.modelos(),
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
        comparacao = estado.adicionar(arquivo.read(), arquivo.filename, formato)
    except ERROS_DE_LEITURA as erro:
        flash(f"O arquivo não foi adicionado. {erro}", "erro")
        return redirect(url_for("dados.tela"))

    repetidas = comparacao.linhas_arquivo - comparacao.linhas_novas
    if not comparacao.linhas_novas:
        flash(f"{comparacao.arquivo} não mudou a base: os {num(repetidas, 0)} acidentes do arquivo já estavam nela.", "aviso")
        return redirect(url_for("dados.tela"))

    flash(f"{comparacao.arquivo} adicionado à base: {num(comparacao.linhas_novas, 0)} acidentes novos"
          + (f" ({num(repetidas, 0)} já estavam na base)." if repetidas else "."), "sucesso")
    return redirect(url_for("dados.tela"))


@bp.post("/remover-adicionados")
def remover_adicionados():
    estado.remover_adicionados()
    flash("Os arquivos adicionados foram removidos. A base voltou a ser a base inicial.", "sucesso")
    return redirect(url_for("dados.tela"))
