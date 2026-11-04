import pandas as pd

AMOSTRA = "data/amostras/acidentes_pr_2024"


def main():
    df = pd.read_csv("data/raw/datatran2024.csv", sep=";", encoding="latin1")

    df = df[df["uf"] == "PR"]

    amostra = df.sample(n=1500, random_state=2903)

    amostra.to_csv(f"{AMOSTRA}.csv", sep=";", encoding="latin1", index=False)

    texto = pd.read_csv(f"{AMOSTRA}.csv", sep=";", encoding="latin1", dtype=str)
    texto.to_json(f"{AMOSTRA}.json", orient="records", force_ascii=False, indent=1)
    texto.to_xml(f"{AMOSTRA}.xml", index=False, root_name="acidentes", row_name="acidente", parser="etree")


if __name__ == "__main__":
    main()
