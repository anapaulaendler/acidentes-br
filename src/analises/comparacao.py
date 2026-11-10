"""Como um lote de dados novos se encaixa na base: o perfil dos acidentes e o desempenho do modelo treinado."""

import pandas as pd

from avaliacao import metricas
from colunas import ALVO

NOMES_METRICAS = {
    "recall": "recall",
    "precisao": "precisão",
    "f1": "F1",
    "precisao_media": "PR-AUC",
    "acuracia": "acurácia",
}


def _letalidade(df: pd.DataFrame, pista: str) -> float:
    selecao = df[df["tipo_pista"] == pista]
    return selecao[ALVO].mean() * 100 if len(selecao) else float("nan")


def perfil(df: pd.DataFrame) -> dict[str, object]:
    return {
        "acidentes": f"{len(df):,}".replace(",", "."),
        "período": f"{df['data_inversa'].min():%d/%m/%Y} a {df['data_inversa'].max():%d/%m/%Y}",
        "UFs": len(df["uf"].dropna().unique()),
        "% fatais": df[ALVO].mean() * 100,
        "% em pista simples": (df["tipo_pista"] == "Simples").mean() * 100,
        "letalidade na pista simples (%)": _letalidade(df, "Simples"),
        "letalidade na pista dupla (%)": _letalidade(df, "Dupla"),
        "% colisão frontal": (df["tipo_acidente"] == "Colisão frontal").mean() * 100,
    }


def comparar_perfil(base: pd.DataFrame, novos: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"base antes do envio": perfil(base), "dados novos": perfil(novos)})


def comparar_modelo(classificador, metricas_validacao: pd.Series, novos: pd.DataFrame) -> pd.DataFrame | None:
    # sem fatais (ou só fatais) o recall e a precisão não dizem nada
    if not novos[ALVO].any() or novos[ALVO].all():
        return None

    probabilidades = classificador.predict_proba(novos)[:, 1]
    tabela = pd.DataFrame({
        "validação cruzada (média)": metricas_validacao,
        "dados novos": metricas(novos[ALVO], probabilidades),
    })
    return tabela.rename(index=NOMES_METRICAS)
