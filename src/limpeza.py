import pandas as pd


# texto que a PRF usa no lugar de um valor faltante
FALTANTES_TEXTO: dict[str, str] = {
    "condicao_metereologica": "Ignorado",
    "sentido_via": "Não Informado",
}

# limites aproximados do território brasileiro (Fernando de Noronha fica de fora, mas não tem BR)
LIMITES_BRASIL = {"latitude": (-34.0, 5.5), "longitude": (-74.0, -34.0)}

# só grafia: o mesmo conceito com texto diferente entre os anos
VARIANTES_CAUSA: dict[str, str] = {
    "Transitar no acostamento": "Transitar no Acostamento",
    "Ingestão de álcool e/ou substâncias psicoativas pelo pedestre": "Pedestre - Ingestão de álcool/ substâncias psicoativas",
    "Ingestão de álcool ou de substâncias psicoativas pelo pedestre": "Pedestre - Ingestão de álcool/ substâncias psicoativas",
}

# a PRF trocou a lista de causas em 2021. os grupos juntam as causas antigas e as novas que descrevem o mesmo
# conceito. desatenção e infração ficam no mesmo grupo porque, em 2021, registros passaram de uma para a outra
GRUPOS_CAUSA: dict[str, tuple[str, ...]] = {
    "Velocidade": (
        "Velocidade Incompatível",
        "Participar de racha",
    ),
    "Álcool e drogas (condutor)": (
        "Ingestão de Álcool",
        "Ingestão de Substâncias Psicoativas",
        "Ingestão de álcool pelo condutor",
        "Ingestão de substâncias psicoativas pelo condutor",
    ),
    "Falha do condutor": (
        "Falta de Atenção à Condução",
        "Reação tardia ou ineficiente do condutor",
        "Ausência de reação do condutor",
        "Condutor usando celular",
        "Frear bruscamente",
        "Acessar a via sem observar a presença dos outros veículos",
        "Desobediência às normas de trânsito pelo condutor",
        "Não guardar distância de segurança",
        "Ultrapassagem Indevida",
        "Condutor deixou de manter distância do veículo da frente",
        "Manobra de mudança de faixa",
        "Conversão proibida",
        "Retorno proibido",
        "Transitar na contramão",
        "Desrespeitar a preferência no cruzamento",
        "Acesso irregular",
        "Transitar no Acostamento",
        "Estacionar ou parar em local proibido",
        "Condutor desrespeitou a iluminação vermelha do semáforo",
        "Trafegar com motocicleta (ou similar) entre as faixas",
        "Transitar na calçada",
        "Deixar de acionar o farol da motocicleta (ou similar)",
        "Condutor não acionou o farol baixo durante o dia em rodovias de pista simples",
    ),
    "Estado físico (condutor)": (
        "Condutor Dormindo",
        "Mal Súbito",
        "Mal súbito do condutor",
        "Suicídio (presumido)",
        "Transtornos Mentais (exceto suicidio)",
    ),
    "Veículo": (
        "Defeito Mecânico no Veículo",
        "Avarias e/ou desgaste excessivo no pneu",
        "Demais falhas mecânicas ou elétricas",
        "Problema com o freio",
        "Problema na suspensão",
        "Deficiência ou não Acionamento do Sistema de Iluminação/Sinalização do Veículo",
        "Deficiência do Sistema de Iluminação/Sinalização",
        "Carga excessiva e/ou mal acondicionada",
        "Faróis desregulados",
        "Modificação proibida",
    ),
    "Via": (
        "Defeito na Via",
        "Sinalização da via insuficiente ou inadequada",
        "Pista esburacada",
        "Obras na pista",
        "Pista em desnível",
        "Demais falhas na via",
        "Ausência de sinalização",
        "Afundamento ou ondulação no pavimento",
        "Acostamento em desnível",
        "Iluminação deficiente",
        "Falta de acostamento",
        "Falta de elemento de contenção que evite a saída do leito carroçável",
        "Sinalização mal posicionada",
        "Obstrução na via",
        "Desvio temporário",
        "Declive acentuado",
        "Curva acentuada",
        "Restrição de visibilidade em curvas horizontais",
        "Restrição de visibilidade em curvas verticais",
        "Faixas de trânsito com largura insuficiente",
        "Semáforo com defeito",
        "Redutor de velocidade em desacordo",
        "Sinalização encoberta",
        "Sistema de drenagem ineficiente",
        "Área urbana sem a presença de local apropriado para a travessia de pedestres",
        "Objeto estático sobre o leito carroçável",
        "Acumulo de areia ou detritos sobre o pavimento",
        "Acumulo de óleo sobre o pavimento",
    ),
    "Clima e natureza": (
        "Fenômenos da Natureza",
        "Pista Escorregadia",
        "Chuva",
        "Acumulo de água sobre o pavimento",
        "Neblina",
        "Fumaça",
        "Demais Fenômenos da natureza",
        "Restrição de Visibilidade",
    ),
    "Pedestre": (
        "Falta de Atenção do Pedestre",
        "Desobediência às normas de trânsito pelo pedestre",
        "Pedestre - Ingestão de álcool/ substâncias psicoativas",
        "Pedestre andava na pista",
        "Entrada inopinada do pedestre",
        "Pedestre cruzava a pista fora da faixa",
    ),
    "Animais": (
        "Animais na Pista",
    ),
    "Agressão externa": (
        "Agressão Externa",
        "Obstrução Via tentativa Assalto",
    ),
}

# tipos que mudaram de nome ou foram divididos em 2021; os outros tipos continuam iguais
GRUPOS_TIPO: dict[str, str] = {
    "Colisão com objeto estático": "Colisão com objeto",
    "Colisão com objeto em movimento": "Colisão com objeto",
    "Colisão lateral mesmo sentido": "Colisão lateral",
    "Colisão lateral sentido oposto": "Colisão lateral",
    "Danos eventuais": "Outros",
    "Eventos atípicos": "Outros",
    "Sinistro pessoal de trânsito": "Outros",
}


def marcar_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for coluna, texto in FALTANTES_TEXTO.items():
        df[coluna] = df[coluna].mask(df[coluna] == texto)

    sem_rodovia = df["br"] == 0
    df["br"] = df["br"].mask(sem_rodovia)
    df["km"] = df["km"].mask(sem_rodovia)

    return df


def padronizar_textos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["causa_acidente"] = df["causa_acidente"].replace(VARIANTES_CAUSA)

    partes = df["tracado_via"].str.split(";")
    df["tracado_via"] = partes.map(lambda p: ";".join(sorted(p)), na_action="ignore")

    return df


def agrupar_categorias(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    causa_para_grupo = {}
    for grupo, causas in GRUPOS_CAUSA.items():
        for causa in causas:
            causa_para_grupo[causa] = grupo

    df["grupo_causa"] = df["causa_acidente"].map(causa_para_grupo)
    df["grupo_tipo"] = df["tipo_acidente"].replace(GRUPOS_TIPO)

    return df


def remover_coordenadas_invalidas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    fora = pd.Series(False, index=df.index)
    for coluna, (minimo, maximo) in LIMITES_BRASIL.items():
        fora |= (df[coluna] < minimo) | (df[coluna] > maximo)

    df.loc[fora, ["latitude", "longitude"]] = float("nan")

    return df


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(ignore_index=True)
    df = marcar_faltantes(df)
    df = padronizar_textos(df)
    df = agrupar_categorias(df)
    df = remover_coordenadas_invalidas(df)

    return df
