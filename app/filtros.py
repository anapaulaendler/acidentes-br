from dataclasses import asdict, dataclass

import pandas as pd

from colunas import ALVO

FATAIS, NAO_FATAIS = "Fatais", "Não fatais"


@dataclass(frozen=True)
class Filtros:
    uf: str | None = None
    ano: str | None = None
    gravidade: str | None = None

    def chave(self) -> tuple:
        return (self.uf, self.ano, self.gravidade)

    def como_dict(self) -> dict[str, str]:
        return {nome: valor for nome, valor in asdict(self).items() if valor}

    def ativos(self) -> bool:
        return bool(self.como_dict())


def ler(args) -> Filtros:
    return Filtros(
        uf=args.get("uf") or None,
        ano=args.get("ano") or None,
        gravidade=args.get("gravidade") or None,
    )


def aplicar(df: pd.DataFrame, filtros: Filtros) -> pd.DataFrame:
    if filtros.uf:
        df = df[df["uf"] == filtros.uf]
    if filtros.ano:
        df = df[df["ano"].astype(str) == filtros.ano]
    if filtros.gravidade == FATAIS:
        df = df[df[ALVO]]
    elif filtros.gravidade == NAO_FATAIS:
        df = df[~df[ALVO]]

    return df


def opcoes(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "uf": sorted(df["uf"].dropna().unique().tolist()),
        "ano": [str(ano) for ano in sorted(df["ano"].dropna().unique())],
        "gravidade": [FATAIS, NAO_FATAIS],
    }
