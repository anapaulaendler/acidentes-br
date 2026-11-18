import pandas as pd
from sklearn.model_selection import train_test_split

from app.estado import ANOS_BASE_COMPLETA, PASTA_RAW, PRODUCAO

TAMANHO = 100


def main() -> None:
    partes = [pd.read_csv(PASTA_RAW / f"datatran{ano}.csv", sep=";", encoding="latin1", dtype=str) for ano in ANOS_BASE_COMPLETA]
    df = pd.concat(partes, ignore_index=True).dropna(subset=["classificacao_acidente"])

    fatal = df["classificacao_acidente"] == "Com Vítimas Fatais"
    producao, _ = train_test_split(df, train_size=TAMANHO, stratify=fatal, random_state=2903)

    producao.to_csv(PRODUCAO, sep=";", encoding="latin1", index=False)
    fatais = (producao["classificacao_acidente"] == "Com Vítimas Fatais").sum()
    print(f"{len(producao)} acidentes ({fatais} fatais) salvos em {PRODUCAO}")


if __name__ == "__main__":
    main()
