import streamlit as st

from estado import exigir_dados

df = exigir_dados()

st.title("A rede neural vê mais?")
