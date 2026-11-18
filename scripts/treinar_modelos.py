"""Treina os modelos com a base completa da PRF (data/raw/) e salva em modelos_treinados/.

Rode da raiz do projeto:
    python3 -m scripts.treinar_modelos                                  # todos os modelos
    python3 -m scripts.treinar_modelos "Rede com one-hot (PyTorch)"     # só esses; os outros vêm do arquivo salvo
"""

import sys
from datetime import datetime
from time import perf_counter

import pandas as pd

import base
from avaliacao import FATIAS, montar_features, validacao_cruzada
from modelos import treinados
from modelos.treino import treinar


def main(nomes: list[str]) -> None:
    inicio = perf_counter()
    # sem a amostra de produção (data/amostras/producao_100.csv): ela fica fora do treino e da validação
    df = base.base_de_treino()
    print(f"{base.NOME}: {len(df):,} acidentes")

    classificadores = {}
    if nomes != treinados.CLASSIFICADORES:
        # só alguns modelos: os outros continuam como estão no arquivo salvo, se a base for a mesma
        anteriores = treinados.carregar()
        if anteriores is not None:
            if anteriores.acidentes != len(df):
                raise SystemExit(f"A base mudou desde o último treino ({anteriores.acidentes:,} → {len(df):,} acidentes): treine todos os modelos.")
            classificadores = dict(anteriores.classificadores)

    for nome in nomes:
        print(f"{nome}: validação cruzada ({FATIAS} fatias)...")
        por_fatia = validacao_cruzada(nome, df)

        # o modelo salvo treina com a base toda; a validação cruzada mede como ele se sai em acidentes que não viu
        print(f"{nome}: treino com a base toda...")
        modelo = treinar(nome, montar_features(df))
        classificadores[nome] = treinados.Classificador(modelo=modelo, metricas_fatias=por_fatia)

        # salva a cada modelo: se o treino cair no meio, os modelos prontos ficam salvos
        treinados.salvar(treinados.ModelosTreinados(
            treinado_em=datetime.now(),
            base=base.NOME,
            acidentes=len(df),
            anos=f"{df['ano'].min()}–{df['ano'].max()}",
            classificadores={n: classificadores[n] for n in treinados.CLASSIFICADORES if n in classificadores},
        ))
        print(pd.DataFrame({"média": por_fatia.mean(), "desvio": por_fatia.std()}).round(3).to_string())

    print(f"Salvo em {treinados.ARQUIVO} ({perf_counter() - inicio:.0f} s)")


if __name__ == "__main__":
    nomes = sys.argv[1:] or treinados.CLASSIFICADORES
    desconhecidos = [nome for nome in nomes if nome not in treinados.CLASSIFICADORES]
    if desconhecidos:
        raise SystemExit(f"Modelos desconhecidos: {desconhecidos}. Opções: {treinados.CLASSIFICADORES}")
    main(nomes)
