"""Treina os modelos com a base da app (base inicial + arquivos adicionados) e salva em modelos_treinados/.

Rode da raiz do projeto:
    python3 -m scripts.treinar_modelos
"""

from datetime import datetime
from time import perf_counter

import pandas as pd

from app import estado
from avaliacao import FATIAS, montar_features, validacao_cruzada
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

    print(f"Validação cruzada do {treinados.CLASSIFICADOR} ({FATIAS} fatias)...")
    por_fatia = validacao_cruzada(treinados.CLASSIFICADOR, df)

    # o modelo salvo treina com a base toda; a validação cruzada mede como ele se sai em acidentes que não viu
    print(f"Treinando o {treinados.CLASSIFICADOR} com a base toda...")
    classificador = treinar(treinados.CLASSIFICADOR, montar_features(df))

    treinados.salvar(treinados.ModelosTreinados(
        treinado_em=datetime.now(),
        base=base.nome,
        acidentes=len(df),
        anos=f"{df['ano'].min()}–{df['ano'].max()}",
        regressao=regressao,
        classificador=classificador,
        metricas_fatias=por_fatia,
        linhas_classificador=len(df),
    ))

    print(f"Salvo em {treinados.ARQUIVO} ({perf_counter() - inicio:.0f} s)")
    print(pd.DataFrame({"média": por_fatia.mean(), "desvio": por_fatia.std()}).round(3).to_string())


if __name__ == "__main__":
    main()
