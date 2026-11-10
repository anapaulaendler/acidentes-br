from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

MINIMO_ACIDENTES = 30

MINIMO_TRECHOS = 50

COLUNAS_PCT = ["pct_pista_simples", "pct_curva", "pct_noite", "pct_chuva", "pct_urbano", "pct_velocidade", "pct_alcool", "pct_pedestre"]

MINIMOS_SENSIBILIDADE = [10, 30, 50, 100]

FATIAS = 5


@dataclass(frozen=True)
class ResultadoRegressao:
    trechos: int
    coeficientes: pd.DataFrame
    metricas: pd.DataFrame
    real: pd.Series  # todos os trechos
    previsto: np.ndarray  # cada trecho previsto pelo modelo da rodada em que ele ficou no teste
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


def _validacao_cruzada(X: pd.DataFrame, y: pd.Series, peso: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cada trecho é previsto pelo modelo que treinou sem ele. Devolve essa previsão e a da base (a média do treino)."""
    previsto = np.empty(len(y))
    media = np.empty(len(y))

    for treino, teste in KFold(n_splits=FATIAS, shuffle=True, random_state=2903).split(X):
        peso_treino = None if peso is None else peso.iloc[treino]
        modelo = _pipeline().fit(X.iloc[treino], y.iloc[treino], regressao__sample_weight=peso_treino)
        previsto[teste] = modelo.predict(X.iloc[teste])
        media[teste] = y.iloc[treino].mean()

    return previsto, media


def _sensibilidade(trechos: pd.DataFrame) -> pd.DataFrame:
    linhas = []

    for minimo in MINIMOS_SENSIBILIDADE:
        selecao = trechos[trechos["acidentes"] >= minimo].reset_index(drop=True)
        if len(selecao) < MINIMO_TRECHOS:
            continue

        X, y, acidentes = _variaveis(selecao), selecao["letalidade_trecho"], selecao["acidentes"]

        for nome_peso, peso in [("igual", None), ("nº de acidentes", acidentes)]:
            previsto, _ = _validacao_cruzada(X, y, peso)
            modelo = _pipeline().fit(X, y, regressao__sample_weight=peso)
            beta = modelo.named_steps["regressao"].coef_[X.columns.get_loc("pct_pista_simples")] if _pista_varia(selecao) else np.nan

            linhas.append({
                "mínimo": minimo,
                "peso": nome_peso,
                "trechos": len(selecao),
                "MAE validação": mean_absolute_error(y, previsto),
                "R² validação": r2_score(y, previsto),
                "R² ponderado validação": r2_score(y, previsto, sample_weight=acidentes),
                "pista simples 0% → 100% (pp)": beta * 100,
            })

    return pd.DataFrame(linhas)


def ajustar(trechos: pd.DataFrame, minimo: int = MINIMO_ACIDENTES) -> ResultadoRegressao | None:
    selecao = trechos[trechos["acidentes"] >= minimo].reset_index(drop=True)
    if len(selecao) < MINIMO_TRECHOS:
        return None

    X = _variaveis(selecao)
    y = selecao["letalidade_trecho"]

    previsto, media = _validacao_cruzada(X, y)
    # o modelo final treina com todos os trechos; a validação cruzada mede como ele se sai em trechos que não viu
    modelo = _pipeline().fit(X, y)

    metricas = pd.DataFrame({
        "treino (todos os trechos)": _metricas(y, modelo.predict(X)),
        "validação cruzada": _metricas(y, previsto),
        "base (média) na validação": _metricas(y, media),
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
        real=y,
        previsto=previsto,
        sensibilidade=_sensibilidade(trechos),
    )
