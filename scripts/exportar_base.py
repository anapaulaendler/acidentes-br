"""Gera os dois arquivos de dados da entrega.

- data/raw/acidentes_prf_brutos_2017_2026.zip: os CSVs da PRF como foram baixados; o notebook roda do zero com ele.
- data/processed/acidentes_prf_2017_2026_limpos.zip: um CSV com a base limpa (a mesma do treino), para usar sem o notebook.

Rode da raiz do projeto: python3 -m scripts.exportar_base
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from base import ANOS, PASTA_RAW, ZIP_BRUTOS, ZIP_LIMPOS, base_de_treino  # noqa: E402


def main() -> None:
    with zipfile.ZipFile(ZIP_BRUTOS, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as arquivo:
        for ano in ANOS:
            arquivo.write(PASTA_RAW / f"datatran{ano}.csv", f"datatran{ano}.csv")
    print(f"{ZIP_BRUTOS}: {ZIP_BRUTOS.stat().st_size / 1e6:.0f} MB")

    ZIP_LIMPOS.parent.mkdir(parents=True, exist_ok=True)
    base_de_treino().to_csv(ZIP_LIMPOS, index=False, compression={
        "method": "zip", "archive_name": ZIP_LIMPOS.name.replace(".zip", ".csv"), "compresslevel": 9})
    print(f"{ZIP_LIMPOS}: {ZIP_LIMPOS.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
