"""Nível 3 da hipótese: o mesmo trecho antes e depois da duplicação (diferença em diferenças)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from colunas import ALVO
from trechos import CHAVE_TRECHO, marcar_trecho

MINIMO_ACIDENTES_ANO = 5   # um ano com menos acidentes não diz qual é a pista do trecho
SIMPLES = 0.7              # ano "simples": pelo menos 70% dos acidentes em pista simples
DUPLA = 0.3                # ano "duplo": no máximo 30%; entre os dois, o ano da obra, que sai da comparação
REPETICOES = 2000


def trechos_por_ano(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por trecho e ano, com os acidentes, os fatais e o estado da pista naquele ano."""
    anos = (marcar_trecho(df).assign(simples=lambda d: d["tipo_pista"].eq("Simples"))
            .groupby(CHAVE_TRECHO + ["ano"])
            .agg(acidentes=(ALVO, "size"), fatais=(ALVO, "sum"), pct_simples=("simples", "mean"))
            .reset_index())
    anos = anos[anos["acidentes"] >= MINIMO_ACIDENTES_ANO].copy()
    anos["estado"] = np.select([anos["pct_simples"] >= SIMPLES, anos["pct_simples"] <= DUPLA], ["S", "D"], "obra")
    return anos


def classificar(anos: pd.DataFrame) -> pd.Series:
    """Duplicou: 2 ou mais anos simples, os anos da obra e 2 ou mais anos duplos, nessa ordem.
    Simples sempre: 4 ou mais anos, todos simples. O sentido contrário (dupla → simples) serve de teste do registro."""
    sequencia = (anos.sort_values("ano").assign(letra=lambda d: d["estado"].replace({"obra": "M"}))
                 .groupby(CHAVE_TRECHO)["letra"].agg("".join))
    return pd.Series(np.select(
        [sequencia.str.fullmatch(r"S{2,}M*D{2,}"), sequencia.str.fullmatch(r"S{4,}"), sequencia.str.fullmatch(r"D{2,}M*S{2,}")],
        ["duplicou", "simples sempre", "dupla → simples"], "outro"), index=sequencia.index, name="grupo")


@dataclass
class Resultado:
    por_trecho: pd.DataFrame   # um trecho duplicado por linha: antes, depois e o controle nos mesmos anos
    resumo: pd.DataFrame       # a estimativa e o intervalo de 95% (bootstrap pelos trechos)
    evento: pd.DataFrame       # a letalidade ano a ano, em relação ao ano da obra
    grupos: pd.Series          # quantos trechos em cada grupo


def _resumo(t: pd.DataFrame) -> dict:
    antes = t["fatais_antes"].sum() / t["acidentes_antes"].sum()
    depois = t["fatais_depois"].sum() / t["acidentes_depois"].sum()
    # o controle de cada trecho, ponderado pelos acidentes dele
    controle_antes = np.average(t["controle_antes"], weights=t["acidentes_antes"])
    controle_depois = np.average(t["controle_depois"], weights=t["acidentes_depois"])
    return {
        "letalidade antes (%)": antes * 100,
        "letalidade depois (%)": depois * 100,
        "controle antes (%)": controle_antes * 100,
        "controle depois (%)": controle_depois * 100,
        "diferença em diferenças (pp)": (depois - antes - (controle_depois - controle_antes)) * 100,
        "redução relativa da letalidade": 1 - (depois / antes) / (controle_depois / controle_antes),
        "fatais por trecho-ano antes": t["fatais_antes"].sum() / t["anos_antes"].sum(),
        "fatais por trecho-ano depois": t["fatais_depois"].sum() / t["anos_depois"].sum(),
    }


def comparar(df: pd.DataFrame, random_state: int = 2903) -> Resultado:
    anos = trechos_por_ano(df)
    grupos = classificar(anos)
    anos = anos.join(grupos, on=CHAVE_TRECHO)

    # o controle: a letalidade de cada ano nos trechos que ficaram em pista simples o período todo
    controle = anos[anos["grupo"] == "simples sempre"].groupby("ano")[["acidentes", "fatais"]].sum()
    controle_ano = controle["fatais"] / controle["acidentes"]

    duplicados = anos[(anos["grupo"] == "duplicou") & (anos["estado"] != "obra")]
    linhas = []
    for chave, trecho in duplicados.groupby(CHAVE_TRECHO):
        linha = dict(zip(CHAVE_TRECHO, chave), ano_duplo=trecho.loc[trecho["estado"] == "D", "ano"].min())
        for lado, estado in [("antes", "S"), ("depois", "D")]:
            parte = trecho[trecho["estado"] == estado]
            linha |= {f"acidentes_{lado}": parte["acidentes"].sum(), f"fatais_{lado}": parte["fatais"].sum(),
                      f"anos_{lado}": len(parte), f"controle_{lado}": np.average(controle_ano[parte["ano"]], weights=parte["acidentes"])}
        linhas.append(linha)
    por_trecho = pd.DataFrame(linhas)

    rng = np.random.default_rng(random_state)
    sorteios = pd.DataFrame([_resumo(por_trecho.iloc[rng.integers(0, len(por_trecho), len(por_trecho))]) for _ in range(REPETICOES)])
    resumo = pd.DataFrame({"estimativa": _resumo(por_trecho),
                           "IC 95%: de": sorteios.quantile(0.025), "IC 95%: até": sorteios.quantile(0.975)})

    # estudo de evento: a diferença para o controle em cada ano relativo à obra (as pontas agrupadas em ≤ −4 e ≥ 4)
    evento = duplicados.merge(por_trecho[CHAVE_TRECHO + ["ano_duplo"]], on=CHAVE_TRECHO)
    evento["ano relativo"] = (evento["ano"] - evento["ano_duplo"]).clip(-4, 4)
    evento["fatais_controle"] = evento["acidentes"] * controle_ano.reindex(evento["ano"]).to_numpy()
    evento = evento.groupby("ano relativo")[["acidentes", "fatais", "fatais_controle"]].sum()
    evento["letalidade (%)"] = evento["fatais"] / evento["acidentes"] * 100
    evento["controle (%)"] = evento["fatais_controle"] / evento["acidentes"] * 100
    evento["diferença (pp)"] = evento["letalidade (%)"] - evento["controle (%)"]

    return Resultado(por_trecho, resumo, evento.drop(columns="fatais_controle"), grupos.value_counts())
