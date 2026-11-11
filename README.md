# Acidentes nas rodovias federais — Equipe 9

Aplicação web (Flask) que analisa a gravidade dos acidentes nas rodovias federais e testa uma hipótese:
**a duplicação de trechos de pista simples reduz a chance de um acidente ser fatal.**

Fonte dos dados: [dados abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf), agrupados **por ocorrência** (`datatran<ano>.csv`).

## Como rodar

1. Instale as dependências (Python 3.11 ou mais recente):

   ```bash
   pip install -r requirements.txt
   ```

2. Para usar a base completa, coloque os arquivos `datatran2017.csv` a `datatran2026.csv` em `data/raw/`.
   Sem esses arquivos, a app abre sem base, na tela **Dados**: o primeiro arquivo enviado vira a base.

3. Inicie a app e abra http://127.0.0.1:5000 no navegador:

   ```bash
   python3 rodar.py
   ```

4. Na tela **Dados**, carregue uma base: envie um arquivo (.csv, .json ou .xml) para adicionar à base.

O mapa precisa de internet (fundo do OpenStreetMap). O resto da app funciona sem internet.

## Telas

| Tela | O que mostra |
|---|---|
| Quando? | A letalidade por hora, dia da semana e mês, e a gravidade geral |
| Onde? | O mapa dos acidentes fatais, as personas dos trechos (K-Means) e a comparação simples × dupla dentro de cada persona |
| Condições | A letalidade por causa, clima e traçado, e a regressão por trecho (coeficientes, qualidade e sensibilidade) |
| Modelos | Um cartão por modelo: a pergunta, os dados, o resultado e o limite |
| Dados | O upload, o relatório da limpeza, a mudança das categorias em 2021 e a tabela dos acidentes |

Cada gráfico fica num **cartão de insight**: a afirmação, a evidência, a leitura e o limite. Os números saem da base carregada.
Uma base nova com as mesmas colunas recalcula todas as telas e todos os modelos.

Os filtros da barra lateral (UF, ano, tipo de pista) valem para todas as telas.

## Estrutura

```
app/                  aplicação Flask
├── telas/            uma rota por tela
├── templates/        HTML (Jinja2); _cartao.html é o cartão de insight
├── estado.py         a base carregada e o cache dos resultados
├── calculos.py       os cálculos de src/ com o cache
├── filtros.py        os filtros da barra lateral
└── graficos.py       os gráficos (Plotly)
src/                  análise e modelos (usados pela app e pelo notebook)
├── carregar.py       leitura de CSV, JSON e XML
├── limpeza.py        limpeza e grupos de causa
├── trechos.py        a tabela de trechos (UF + BR + faixa de 5 km)
├── analises/         letalidade e relatório da limpeza
└── modelos/          regressão por trecho, personas (K-Means), classificadores
notebooks/analise_prf.ipynb   a análise completa de Data Science
tests/                testes (pytest)
```

## Testes

```bash
python3 -m pytest
```

Os testes que precisam da base completa são ignorados quando `data/raw/` está vazia.
