import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from colunas import ALVO
from trechos import CHAVE_TRECHO, TAMANHO_FAIXA_KM, montar_trechos

MINIMO_ACIDENTES = 30
K = 4

MINIMO_TRECHOS = 40

COLUNAS_CLUSTER = ["acidentes_por_ano", "pct_noite", "pct_chuva", "pct_curva", "pct_fim_de_semana", "media_veiculos", "media_pessoas"]

# contagens muito assimétricas: o log evita que poucos trechos movimentados dominem a distância
COLUNAS_LOG = ["acidentes_por_ano", "media_veiculos", "media_pessoas"]

# cada persona é o cluster com o maior valor (padronizado) nesta coluna
REGRAS_PERSONA: dict[str, str] = {
    "Corredores urbanos movimentados": "acidentes_por_ano",
    "Serra com curva e chuva": "pct_curva",
    "Rurais com acidentes de vários veículos": "media_pessoas",
    "Noite e fim de semana": "pct_noite",
}

ACOES: dict[str, str] = {
    "Corredores urbanos movimentados": "Fiscalização de velocidade e travessias de pedestres. A duplicação tem pouco efeito.",
    "Serra com curva e chuva": "Prioridade de duplicação. Também: drenagem, sinalização de curva e defensas.",
    "Rurais com acidentes de vários veículos": "Prioridade de duplicação. Também: faixas adicionais de ultrapassagem.",
    "Noite e fim de semana": "Prioridade de duplicação. Também: iluminação e fiscalização noturna de álcool.",
}


def _padronizar(trechos: pd.DataFrame) -> np.ndarray:
    X = trechos[COLUNAS_CLUSTER].astype(float)
    X[COLUNAS_LOG] = np.log(X[COLUNAS_LOG])

    return StandardScaler().fit_transform(X.fillna(X.median()))


def _selecionar(trechos: pd.DataFrame, minimo: int) -> pd.DataFrame:
    return trechos[trechos["acidentes"] >= minimo].reset_index(drop=True)


def nomear_clusters(centros: pd.DataFrame) -> dict[int, str]:
    nomes = {}
    for persona, coluna in REGRAS_PERSONA.items():
        cluster = int(centros[coluna].idxmax())
        # com outra base, duas regras podem escolher o mesmo cluster; a segunda fica sem efeito
        if cluster not in nomes:
            nomes[cluster] = persona

    for cluster in centros.index:
        nomes.setdefault(int(cluster), f"Grupo {int(cluster) + 1}")

    return nomes


def diagnostico_k(trechos: pd.DataFrame, minimo: int = MINIMO_ACIDENTES) -> pd.DataFrame | None:
    selecao = _selecionar(trechos, minimo)
    if len(selecao) < MINIMO_TRECHOS:
        return None

    Z = _padronizar(selecao)
    linhas = []
    for k in range(2, 11):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)
        linhas.append({"K": k, "inércia": modelo.inertia_, "silhueta": silhouette_score(Z, modelo.labels_)})

    return pd.DataFrame(linhas).set_index("K")


def agrupar(trechos: pd.DataFrame, minimo: int = MINIMO_ACIDENTES, k: int = K) -> pd.DataFrame | None:
    selecao = _selecionar(trechos, minimo)
    if len(selecao) < MINIMO_TRECHOS:
        return None

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(_padronizar(selecao))
    centros = pd.DataFrame(kmeans.cluster_centers_, columns=COLUNAS_CLUSTER)

    selecao["cluster"] = kmeans.labels_
    selecao["persona"] = selecao["cluster"].map(nomear_clusters(centros))

    return selecao


def perfil(agrupados: pd.DataFrame) -> pd.DataFrame:
    tabela = agrupados.groupby("persona").agg(
        trechos=("acidentes", "size"),
        **{coluna: (coluna, "mean") for coluna in COLUNAS_CLUSTER},
        pct_urbano=("pct_urbano", "mean"),
        pct_pista_simples=("pct_pista_simples", "mean"),
        letalidade_media=("letalidade_trecho", "mean"),
    )
    tabela["ação sugerida"] = tabela.index.map(lambda persona: ACOES.get(persona, "Sem ação sugerida: o grupo não segue nenhuma regra de persona."))

    return tabela.sort_values("letalidade_media", ascending=False)


def comparar_pistas(agrupados: pd.DataFrame) -> pd.DataFrame:
    # "simples" se pelo menos 80% dos acidentes são em pista simples; os trechos mistos saem da comparação
    pista = pd.Series(np.select(
        [agrupados["pct_pista_simples"] >= 80, agrupados["pct_pista_simples"] <= 20],
        ["simples", "dupla ou múltipla"],
        "misto",
    ), index=agrupados.index)
    fatais = agrupados["letalidade_trecho"] / 100 * agrupados["acidentes"]

    base = agrupados.assign(pista_predominante=pista, fatais=fatais)
    base = base[base["pista_predominante"] != "misto"]

    # letalidade do conjunto: cada acidente tem o mesmo peso
    soma = base.groupby(["persona", "pista_predominante"]).agg(
        trechos=("acidentes", "size"), acidentes=("acidentes", "sum"), fatais=("fatais", "sum")
    )
    # com um filtro de pista, um dos lados pode não existir: a coluna fica com NaN, mas existe
    lados = ["simples", "dupla ou múltipla"]
    letalidade = (soma["fatais"] / soma["acidentes"] * 100).unstack("pista_predominante").reindex(columns=lados)
    trechos = soma["trechos"].unstack("pista_predominante").reindex(columns=lados)

    tabela = pd.DataFrame({
        "dupla ou múltipla": letalidade["dupla ou múltipla"],
        "simples": letalidade["simples"],
        "trechos simples": trechos["simples"],
        "trechos dupla ou múltipla": trechos["dupla ou múltipla"],
    }).astype(float)
    tabela["diferença (pp)"] = tabela["simples"] - tabela["dupla ou múltipla"]

    return tabela.sort_values("diferença (pp)", ascending=False)


SEM_PERSONA = "Sem persona"


class MarcarPersona(BaseEstimator, TransformerMixin):
    """Acrescenta a coluna "persona": a persona do K-Means do trecho (UF, BR, faixa de 5 km) de cada acidente.

    O fit monta os trechos e roda o K-Means só com os acidentes que recebe (o treino, ou as fatias de treino da
    validação cruzada). Sem o alvo: os trechos usam só as condições dos acidentes. O transform só consulta a tabela,
    então um acidente do teste não muda as personas. Um trecho com menos de 30 acidentes no treino fica "Sem persona".
    """

    def fit(self, X: pd.DataFrame, y=None) -> "MarcarPersona":
        trechos = montar_trechos(X.drop(columns=[ALVO], errors="ignore"))
        agrupados = agrupar(trechos)
        if agrupados is None:
            self.persona_do_trecho_ = pd.Series(dtype=str)
        else:
            self.persona_do_trecho_ = agrupados.set_index(CHAVE_TRECHO)["persona"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        faixa = X["km"] // TAMANHO_FAIXA_KM * TAMANHO_FAIXA_KM
        chaves = pd.MultiIndex.from_arrays([X["uf"], X["br"], faixa], names=CHAVE_TRECHO)
        persona = self.persona_do_trecho_.reindex(chaves).fillna(SEM_PERSONA).to_numpy()
        return X.assign(persona=persona)
