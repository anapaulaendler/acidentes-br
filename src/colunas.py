OBRIGATORIAS: tuple[str, ...] = (
    "data_inversa",
    "dia_semana",
    "horario",
    "uf",
    "br",
    "km",
    "municipio",
    "causa_acidente",
    "tipo_acidente",
    "classificacao_acidente",
    "fase_dia",
    "sentido_via",
    "condicao_metereologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
    "pessoas",
    "veiculos",
    "latitude",
    "longitude",
)

# as colunas que poderiam servir como colinha para o modelo
VAZAMENTO: tuple[str, ...] = (
    "classificacao_acidente",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "feridos",
    "ilesos",
    "ignorados" # do dicionário de dados: "Total de pessoas envolvidas na ocorrência e que não se soube o estado físico"
)

ADMINISTRATIVAS: tuple[str, ...] = (
    "id",
    "regional",
    "delegacia",
    "uop" # do dicionário de dados: "UOP= unidade operacional. Unidade operacional da PRF cujo acidente ocorreu dentro dos limites de sua circunscrição."
)

ALVO = "fatal"

CATEGORICAS: tuple[str, ...] = (
    "uf",
    "br",
    "dia_semana",
    "causa_acidente",
    "tipo_acidente",
    "fase_dia",
    "sentido_via",
    "condicao_metereologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo"
)

NUMERICAS: tuple[str, ...] = (
    "km",
    "pessoas",
    "veiculos"
)
