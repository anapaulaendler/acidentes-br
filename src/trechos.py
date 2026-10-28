import pandas as pd

from colunas import ALVO


TAMANHO_FAIXA_KM = 5

REGIOES: dict[str, tuple[str, ...]] = {
    "Norte": ("AC", "AM", "AP", "PA", "RO", "RR", "TO"),
    "Nordeste": ("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"),
    "Centro-Oeste": ("DF", "GO", "MS", "MT"),
    "Sudeste": ("ES", "MG", "RJ", "SP"),
    "Sul": ("PR", "RS", "SC"),
}

CHAVE_TRECHO = ["uf", "br", "faixa_km"]


def marcar_trecho(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["br", "km"]).copy()
    df["faixa_km"] = (df["km"] // TAMANHO_FAIXA_KM * TAMANHO_FAIXA_KM).astype(int)

    return df


def montar_trechos(df: pd.DataFrame) -> pd.DataFrame:
    df = marcar_trecho(df)
    anos = df["data_inversa"].dt.year.nunique()

    # o clima tem NaN, e um acidente sem clima não conta nem como chuva nem como tempo seco
    chuva = df["condicao_metereologica"].isin(["Chuva", "Garoa/Chuvisco"])
    chuva = chuva.where(df["condicao_metereologica"].notna())

    df = df.assign(
        simples=df["tipo_pista"] == "Simples",
        curva=df["tracado_via"].str.contains("Curva"),
        noite=df["fase_dia"] == "Plena Noite",
        fim_de_semana=df["dia_semana"].isin(["sábado", "domingo"]),
        chuva=chuva,
        urbano=df["uso_solo"] == "Sim",
        velocidade=df["grupo_causa"] == "Velocidade",
        alcool=df["grupo_causa"] == "Álcool e drogas (condutor)",
        pedestre=df["grupo_causa"] == "Pedestre",
    )

    trechos = df.groupby(CHAVE_TRECHO).agg(
        acidentes=(ALVO, "size"),
        letalidade_trecho=(ALVO, "mean"),
        pct_pista_simples=("simples", "mean"),
        pct_curva=("curva", "mean"),
        pct_noite=("noite", "mean"),
        pct_fim_de_semana=("fim_de_semana", "mean"),
        pct_chuva=("chuva", "mean"),
        pct_urbano=("urbano", "mean"),
        pct_velocidade=("velocidade", "mean"),
        pct_alcool=("alcool", "mean"),
        pct_pedestre=("pedestre", "mean"),
        media_veiculos=("veiculos", "mean"),
        media_pessoas=("pessoas", "mean"),
    ).reset_index()

    # taxas em pontos percentuais (0 a 100), para o coeficiente da regressão ser lido em pp
    percentuais = ["letalidade_trecho"] + [coluna for coluna in trechos.columns if coluna.startswith("pct_")]
    trechos[percentuais] = trechos[percentuais] * 100

    trechos["acidentes_por_ano"] = trechos["acidentes"] / anos

    uf_para_regiao = {}
    for regiao, ufs in REGIOES.items():
        for uf in ufs:
            uf_para_regiao[uf] = regiao

    trechos["regiao"] = trechos["uf"].map(uf_para_regiao)

    return trechos
