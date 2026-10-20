import pandas as pd

def main():
    df = pd.read_csv("data/raw/datatran2024.csv", sep=";", encoding="latin1")

    df = df[df["uf"] == "PR"]

    amostra = df.sample(n=1500, random_state=2903)

    amostra.to_csv("data/amostras/acidentes_pr_2024.csv", sep=";", encoding="latin1", index=False)


if __name__ == "__main__":
    main()