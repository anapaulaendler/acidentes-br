from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class Filtros:
    uf: str | None = None
    ano: str | None = None
    pista: str | None = None

    def chave(self) -> tuple:
        return (self.uf, self.ano, self.pista)

    def como_dict(self) -> dict[str, str]:
        return {nome: valor for nome, valor in asdict(self).items() if valor}

    def ativos(self) -> bool:
        return bool(self.como_dict())


def ler(args) -> Filtros:
    return Filtros(
        uf=args.get("uf") or None,
        ano=args.get("ano") or None,
        pista=args.get("pista") or None,
    )


def aplicar(df: pd.DataFrame, filtros: Filtros) -> pd.DataFrame:
    if filtros.uf:
        df = df[df["uf"] == filtros.uf]
    if filtros.ano:
        df = df[df["ano"].astype(str) == filtros.ano]
    if filtros.pista:
        df = df[df["tipo_pista"] == filtros.pista]

    return df


def opcoes(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "uf": sorted(df["uf"].dropna().unique().tolist()),
        "ano": [str(ano) for ano in sorted(df["ano"].dropna().unique())],
        "pista": sorted(df["tipo_pista"].dropna().unique().tolist()),
    }
