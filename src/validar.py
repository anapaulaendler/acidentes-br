import pandas as pd

from colunas import OBRIGATORIAS


class ErroDeColunas(Exception):
    def __init__(self, faltando: list[str]):
        self.faltando = faltando
        super().__init__(f"Faltam as colunas: {','.join(faltando)}") 


def validar_colunas(df: pd.DataFrame):
    faltando = []
    colunas_df = df.columns.tolist()

    for coluna in OBRIGATORIAS:
        if coluna not in colunas_df:
            faltando.append(coluna)

    if faltando:
        raise ErroDeColunas(faltando=faltando)
