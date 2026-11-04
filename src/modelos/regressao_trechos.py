from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MINIMO_ACIDENTES = 30

MINIMO_TRECHOS = 50

COLUNAS_PCT = ["pct_pista_simples", "pct_curva", "pct_noite", "pct_chuva", "pct_urbano", "pct_velocidade", "pct_alcool", "pct_pedestre"]

MINIMOS_SENSIBILIDADE = [10, 30, 50, 100]


@dataclass(frozen=True)
class ResultadoRegressao:
    trechos: int
    coeficientes: pd.DataFrame
    metricas: pd.DataFrame
    real: pd.Series
    previsto: np.ndarray
    sensibilidade: pd.DataFrame


def _variaveis(trechos: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(trechos[COLUNAS_PCT + ["acidentes_por_ano", "regiao"]], columns=["regiao"], drop_first=True, dtype=float)


def _pipeline() -> Pipeline:
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("regressao", LinearRegression()),
    ])


def _metricas(y_real, y_previsto) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_real, y_previsto),
        "RMSE": mean_squared_error(y_real, y_previsto) ** 0.5,
        "R²": r2_score(y_real, y_previsto),
    }


def _pista_varia(trechos: pd.DataFrame) -> bool:
    return trechos["pct_pista_simples"].nunique() > 1


def _sensibilidade(trechos: pd.DataFrame) -> pd.DataFrame:
    linhas = []

    for minimo in MINIMOS_SENSIBILIDADE:
        selecao = trechos[trechos["acidentes"] >= minimo].reset_index(drop=True)
        if len(selecao) < MINIMO_TRECHOS:
            continue

        X = _variaveis(selecao)
        X_tr, X_te, y_tr, y_te, peso_tr, peso_te = train_test_split(
            X, selecao["letalidade_trecho"], selecao["acidentes"], test_size=0.2, random_state=2903
        )

        for nome_peso, peso in [("igual", None), ("nº de acidentes", peso_tr)]:
            modelo = _pipeline().fit(X_tr, y_tr, regressao__sample_weight=peso)
            previsto = modelo.predict(X_te)
            beta = modelo.named_steps["regressao"].coef_[X.columns.get_loc("pct_pista_simples")] if _pista_varia(selecao) else np.nan

            linhas.append({
                "mínimo": minimo,
                "peso": nome_peso,
                "trechos": len(selecao),
                "MAE teste": mean_absolute_error(y_te, previsto),
                "R² teste": r2_score(y_te, previsto),
                "R² ponderado teste": r2_score(y_te, previsto, sample_weight=peso_te),
                "pista simples 0% → 100% (pp)": beta * 100,
            })

    return pd.DataFrame(linhas)


def ajustar(trechos: pd.DataFrame, minimo: int = MINIMO_ACIDENTES) -> ResultadoRegressao | None:
    selecao = trechos[trechos["acidentes"] >= minimo].reset_index(drop=True)
    if len(selecao) < MINIMO_TRECHOS:
        return None

    X = _variaveis(selecao)
    y = selecao["letalidade_trecho"]
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=2903)

    modelo = _pipeline().fit(X_treino, y_treino)
    previsto = modelo.predict(X_teste)

    metricas = pd.DataFrame({
        "treino": _metricas(y_treino, modelo.predict(X_treino)),
        "teste": _metricas(y_teste, previsto),
        "base (média) no teste": _metricas(y_teste, np.full(len(y_teste), y_treino.mean())),
    }).T

    coeficientes = pd.DataFrame({"β": modelo.named_steps["regressao"].coef_}, index=X.columns)
    # o efeito de 0% a 100% só faz sentido nas variáveis em porcentagem
    coeficientes["efeito de 0% → 100%"] = (coeficientes["β"] * 100).where(coeficientes.index.str.startswith("pct_"))
    if not _pista_varia(selecao):
        coeficientes.loc["pct_pista_simples", ["β", "efeito de 0% → 100%"]] = np.nan

    return ResultadoRegressao(
        trechos=len(selecao),
        coeficientes=coeficientes,
        metricas=metricas,
        real=y_teste,
        previsto=previsto,
        sensibilidade=_sensibilidade(trechos),
    )
