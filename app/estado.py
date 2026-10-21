import pandas as pd
import streamlit as st

CHAVE_DADOS = "dados"


def exigir_dados() -> pd.DataFrame:
    if CHAVE_DADOS not in st.session_state:
        st.info("Envie uma base na tela Início.")
        st.stop()

    return st.session_state[CHAVE_DADOS]
