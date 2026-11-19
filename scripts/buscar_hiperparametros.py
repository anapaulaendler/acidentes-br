"""Busca os hiperparâmetros dos modelos do scikit-learn com RandomizedSearchCV e salva em modelos_treinados/busca.joblib.

A busca usa uma amostra estratificada do treino (o teste fica de fora) e 3 fatias. O treino completo
(scripts/treinar_modelos.py) usa a configuração escolhida aqui.

Rode da raiz do projeto:
    python3 -m scripts.buscar_hiperparametros                        # todos os modelos com espaço de busca
    python3 -m scripts.buscar_hiperparametros "Gradient Boosting"    # só esses; os outros continuam no arquivo
"""

import sys
from time import perf_counter

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split

import base
from avaliacao import dividir_treino_teste, montar_features
from colunas import ALVO
from modelos import busca
from modelos.treino import montar

AMOSTRA = 150_000
FATIAS = 3
# o Random Forest e o KNN já usam todos os núcleos; nos outros, as fatias rodam em paralelo
EM_PARALELO = {"Random Forest": 1, "KNN": 1}


def main(nomes: list[str]) -> None:
    treino, _ = dividir_treino_teste(base.base_de_treino())
    amostra, _ = train_test_split(treino, train_size=AMOSTRA, stratify=treino[ALVO], random_state=2903)
    X, y = montar_features(amostra), amostra[ALVO]
    fatias = StratifiedKFold(n_splits=FATIAS, shuffle=True, random_state=2903)
    print(f"Amostra do treino: {len(amostra):,} acidentes, {FATIAS} fatias.")

    resultados = busca.carregar()
    for nome in nomes:
        inicio = perf_counter()
        espaco, combinacoes = busca.ESPACOS[nome]
        n_jobs = EM_PARALELO.get(nome, FATIAS)

        print(f"{nome}: {combinacoes} combinações...", flush=True)
        pesquisa = RandomizedSearchCV(
            montar(nome), espaco, n_iter=combinacoes, scoring=busca.METRICA, cv=fatias,
            refit=False, random_state=2903, n_jobs=n_jobs, error_score="raise",
        ).fit(X, y)
        padrao = cross_val_score(montar(nome), X, y, scoring=busca.METRICA, cv=fatias, n_jobs=n_jobs).mean()

        tabela = pd.DataFrame(pesquisa.cv_results_)
        colunas = [c for c in tabela.columns if c.startswith("param_")]
        tabela = tabela[colunas + ["mean_test_score", "std_test_score", "mean_fit_time"]]
        tabela = tabela.rename(columns=lambda c: c.removeprefix("param_modelo__")).sort_values("mean_test_score", ascending=False)

        melhor = pesquisa.best_score_
        resultados[nome] = busca.ResultadoBusca(
            melhores=pesquisa.best_params_,
            pr_auc_melhores=melhor,
            pr_auc_padrao=padrao,
            escolhidos=pesquisa.best_params_ if melhor > padrao else {},
            resultados=tabela.reset_index(drop=True),
            acidentes=len(amostra),
            segundos=perf_counter() - inicio,
        )
        busca.salvar(resultados)
        print(f"  padrão: PR-AUC {padrao:.4f}. Melhor sorteada: {melhor:.4f} {pesquisa.best_params_} "
              f"({perf_counter() - inicio:.0f} s)", flush=True)


if __name__ == "__main__":
    nomes = sys.argv[1:] or list(busca.ESPACOS)
    desconhecidos = [nome for nome in nomes if nome not in busca.ESPACOS]
    if desconhecidos:
        raise SystemExit(f"Sem espaço de busca: {desconhecidos}. Opções: {list(busca.ESPACOS)}")
    main(nomes)
