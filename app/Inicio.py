from pathlib import Path

import streamlit as st

from carregar import carregar
from estado import CHAVE_DADOS
from resumo import resumo_base
from validar import ErroDeColunas

AMOSTRA = Path(__file__).resolve().parent.parent / "data" / "amostras" / "acidentes_pr_2024.csv"

st.title("Acidentes nas rodovias federais")

arquivo = st.file_uploader("Envie uma base da PRF", type="csv")
usar_amostra = st.button("Usar amostra (PR 2024)")

fonte = None
if usar_amostra:
    fonte = AMOSTRA
elif arquivo is not None:
    fonte = arquivo

if fonte is not None:
    try:
        st.session_state[CHAVE_DADOS] = carregar(fonte)
    except ErroDeColunas as e:
        st.error(f"Faltam as colunas: {', '.join(e.faltando)}")

if CHAVE_DADOS in st.session_state:
    resumo = resumo_base(st.session_state[CHAVE_DADOS])

    col_linhas, col_fatal = st.columns(2)
    col_linhas.metric("Acidentes", resumo["linhas"])
    col_fatal.metric("Fatais", f"{resumo['pct_fatal']:.1f}%")

    st.write(f"**Anos:** {', '.join(str(ano) for ano in resumo['anos'])}")
    st.write(f"**UFs:** {', '.join(resumo['ufs'])}")
