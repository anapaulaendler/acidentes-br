"""Treina os modelos com a base da app (base inicial + arquivos adicionados) e salva em modelos_treinados/.

Rode da raiz do projeto:
    python3 -m scripts.treinar_modelos
"""

from datetime import datetime
from time import perf_counter

import pandas as pd

from app import estado
from avaliacao import dividir_treino_teste, metricas
from colunas import ALVO
from modelos import regressao_trechos, treinados
from modelos.treino import treinar
from trechos import montar_trechos


def main() -> None:
    inicio = perf_counter()
    base = estado.iniciar(com_modelos=False)
    df = base.limpo
    print(f"{base.nome} + {base.arquivos_adicionados} arquivo(s) adicionado(s): {len(df):,} acidentes")

    print("Treinando a regressão por trecho...")
    regressao = regressao_trechos.ajustar(montar_trechos(df))

    print(f"Treinando o {treinados.CLASSIFICADOR}...")
    divisao = dividir_treino_teste(df)
    classificador = treinar(treinados.CLASSIFICADOR, divisao.treino)
    resultado = metricas(divisao.teste[ALVO], classificador.predict_proba(divisao.teste)[:, 1])

    treinados.salvar(treinados.ModelosTreinados(
        treinado_em=datetime.now(),
        base=base.nome,
        acidentes=len(df),
        anos=f"{df['ano'].min()}–{df['ano'].max()}",
        regressao=regressao,
        classificador=classificador,
        metricas_classificador=pd.Series(resultado),
        linhas_classificador=len(df),
    ))

    print(f"Salvo em {treinados.ARQUIVO} ({perf_counter() - inicio:.0f} s)")
    print(pd.Series(resultado).round(3).to_string())


if __name__ == "__main__":
    main()
