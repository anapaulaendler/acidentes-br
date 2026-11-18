"""Treina os modelos com a base completa da PRF (data/raw/) e salva em modelos_treinados/.

A base é dividida em 80% de treino e 20% de teste, estratificado. A validação cruzada roda só no treino;
o modelo final treina com o treino todo e é medido uma vez no teste.

Rode da raiz do projeto:
    python3 -m scripts.treinar_modelos                                  # todos os modelos
    python3 -m scripts.treinar_modelos "Rede com one-hot (PyTorch)"     # só esses; os outros vêm do arquivo salvo
"""

import sys
from datetime import datetime
from time import perf_counter

import pandas as pd

import base
from avaliacao import FATIAS, amostra_de_treino, dividir_treino_teste, metricas, montar_features, validacao_cruzada
from colunas import ALVO
from modelos import treinados
from modelos.treino import treinar


def main(nomes: list[str]) -> None:
    inicio = perf_counter()
    # sem a amostra de produção (data/amostras/producao_100.csv): ela fica fora do treino e da validação
    df = base.base_de_treino()
    print(f"{base.NOME}: {len(df):,} acidentes")

    # 80% para o treino (com a validação cruzada dentro dele) e 20% para o teste final
    treino, teste = dividir_treino_teste(df)
    amostra = amostra_de_treino(treino)
    print(f"Treino: {len(treino):,} acidentes. Teste: {len(teste):,} acidentes.")

    classificadores = {}
    if nomes != treinados.CLASSIFICADORES:
        # só alguns modelos: os outros continuam como estão no arquivo salvo, se a base e a divisão forem as mesmas
        anteriores = treinados.carregar()
        if anteriores is not None:
            if getattr(anteriores, "acidentes_treino", None) != len(treino) or anteriores.acidentes != len(df):
                raise SystemExit("A base ou a divisão treino/teste mudou desde o último treino: treine todos os modelos.")
            classificadores = dict(anteriores.classificadores)

    for nome in nomes:
        print(f"{nome}: validação cruzada no treino ({FATIAS} fatias)...")
        por_fatia = validacao_cruzada(nome, treino)

        print(f"{nome}: treino final e teste...")
        modelo = treinar(nome, montar_features(treino))
        probabilidades = modelo.predict_proba(montar_features(teste))[:, 1]
        classificadores[nome] = treinados.Classificador(
            modelo=modelo,
            metricas_fatias=por_fatia,
            metricas_treino=metricas(amostra[ALVO], modelo.predict_proba(montar_features(amostra))[:, 1]),
            metricas_teste=metricas(teste[ALVO], probabilidades),
            probabilidades_teste=probabilidades.astype("float32"),
        )

        # salva a cada modelo: se o treino cair no meio, os modelos prontos ficam salvos
        treinados.salvar(treinados.ModelosTreinados(
            treinado_em=datetime.now(),
            base=base.NOME,
            acidentes=len(df),
            anos=f"{df['ano'].min()}–{df['ano'].max()}",
            acidentes_treino=len(treino),
            alvo_teste=teste[ALVO].to_numpy(),
            classificadores={n: classificadores[n] for n in treinados.CLASSIFICADORES if n in classificadores},
        ))
        resultado = pd.DataFrame({
            "validação (média)": por_fatia.mean(),
            "validação (desvio)": por_fatia.std(),
            "treino": classificadores[nome].metricas_treino,
            "teste": classificadores[nome].metricas_teste,
        })
        print(resultado.round(3).to_string(), flush=True)

    print(f"Salvo em {treinados.ARQUIVO} ({perf_counter() - inicio:.0f} s)")


if __name__ == "__main__":
    nomes = sys.argv[1:] or treinados.CLASSIFICADORES
    desconhecidos = [nome for nome in nomes if nome not in treinados.CLASSIFICADORES]
    if desconhecidos:
        raise SystemExit(f"Modelos desconhecidos: {desconhecidos}. Opções: {treinados.CLASSIFICADORES}")
    main(nomes)
