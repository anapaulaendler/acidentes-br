"""Treina os modelos com a base da app (base inicial + arquivos adicionados) e salva em modelos_treinados/.

Rode da raiz do projeto:
    python3 -m scripts.treinar_modelos
"""

from datetime import datetime
from time import perf_counter

import pandas as pd

from app import estado
from avaliacao import FATIAS, montar_features, validacao_cruzada
from modelos import treinados
from modelos.treino import treinar


def main() -> None:
    inicio = perf_counter()
    # sem a amostra de produção (data/amostras/producao_100.csv): ela fica fora do treino e da validação
    base = estado.iniciar(com_modelos=False, para_treino=True)
    if base is None:
        raise SystemExit("Sem base para treinar: coloque os datatran*.csv em data/raw/ ou envie um arquivo pela tela Dados.")
    df = base.limpo
    print(f"{base.nome} + {base.arquivos_adicionados} arquivo(s) adicionado(s): {len(df):,} acidentes")

    classificadores = {}
    for nome in treinados.CLASSIFICADORES:
        print(f"{nome}: validação cruzada ({FATIAS} fatias)...")
        por_fatia = validacao_cruzada(nome, df)

        # o modelo salvo treina com a base toda; a validação cruzada mede como ele se sai em acidentes que não viu
        print(f"{nome}: treino com a base toda...")
        modelo = treinar(nome, montar_features(df))
        classificadores[nome] = treinados.Classificador(modelo=modelo, metricas_fatias=por_fatia)

        # salva a cada modelo: se o treino cair no meio, os modelos prontos ficam salvos
        treinados.salvar(treinados.ModelosTreinados(
            treinado_em=datetime.now(),
            base=base.nome,
            acidentes=len(df),
            anos=f"{df['ano'].min()}–{df['ano'].max()}",
            classificadores=classificadores,
        ))
        print(pd.DataFrame({"média": por_fatia.mean(), "desvio": por_fatia.std()}).round(3).to_string())

    print(f"Salvo em {treinados.ARQUIVO} ({perf_counter() - inicio:.0f} s)")


if __name__ == "__main__":
    main()
