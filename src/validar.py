import pandas as pd

from colunas import OBRIGATORIAS


class ErroDeColunas(Exception):
    def __init__(self, faltando: list[str]):
        self.faltando = faltando
        super().__init__(f"Faltam as colunas: {','.join(faltando)}") 


class ErroDeFormato(Exception):
    def __init__(self, extensao: str):
        self.extensao = extensao
        super().__init__(f"Formato {extensao or 'sem extensão'} não aceito. Envie .csv, .json ou .xml.")


def validar_colunas(df: pd.DataFrame):
    faltando = []
    colunas_df = df.columns.tolist()

    for coluna in OBRIGATORIAS:
        if coluna not in colunas_df:
            faltando.append(coluna)

    if faltando:
        raise ErroDeColunas(faltando=faltando)
