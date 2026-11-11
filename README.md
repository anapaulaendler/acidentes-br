# Acidentes nas rodovias federais — Equipe 9

Aplicação web (Flask) que mostra os acidentes nas rodovias federais e testa uma pergunta:
**um modelo consegue prever se um acidente vai ser fatal, só com as condições do acidente?**

Fonte dos dados: [dados abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf), agrupados **por ocorrência** (`datatran<ano>.csv`).

## Como rodar

1. Instale as dependências (Python 3.11 ou mais recente):

   ```bash
   pip install -r requirements.txt
   ```

2. Para usar a base completa, coloque os arquivos `datatran2017.csv` a `datatran2026.csv` em `data/raw/`.
   Sem esses arquivos, a app abre sem base, na tela **Dados adicionados**: o primeiro arquivo enviado vira a base.

3. Inicie a app e abra http://127.0.0.1:5000 no navegador:

   ```bash
   python3 rodar.py
   ```

4. Para treinar os modelos de novo (por exemplo, depois de adicionar dados), rode na raiz do projeto e reinicie a app:

   ```bash
   python3 -m scripts.treinar_modelos
   ```

O mapa precisa de internet (fundo do OpenStreetMap). O resto da app funciona sem internet.

## Telas

| Tela | O que mostra |
|---|---|
| Acidentes | O mapa dos acidentes, paginado, e os acidentes por UF (fatais e não fatais) |
| Modelos | O desempenho dos classificadores na validação cruzada, a rede com embeddings e a previsão dos dados adicionados, acidente por acidente |
| Dados | A lista paginada de todos os acidentes da base |
| Dados adicionados | O envio de arquivos e a lista paginada só com os acidentes enviados |

Cada resultado fica num **cartão de insight**: a afirmação, a evidência, a leitura e o limite.
Um arquivo enviado entra na base e muda as telas na hora. Os modelos só mudam com um novo treino.

Os filtros da barra lateral (acidentes fatais ou não, UF e ano) valem para o mapa, o gráfico por UF e as listas.

## Estrutura

```
app/                  aplicação Flask
├── telas/            uma rota por tela
├── templates/        HTML (Jinja2); _cartao.html é o cartão de insight, _lista.html a lista paginada
├── static/           o CSS
├── estado.py         a base carregada, os dados adicionados, os modelos e a previsão
├── filtros.py        os filtros da barra lateral
└── formato.py        números e tabelas no formato brasileiro
src/                  dados e modelos
├── carregar.py       leitura de CSV, JSON e XML
├── validar.py        as colunas obrigatórias
├── limpeza.py        limpeza e grupos de causa
├── colunas.py        os grupos de colunas e o alvo (fatal)
├── avaliacao.py      a validação cruzada e as métricas
├── features/         a codificação das colunas
└── modelos/          árvore de decisão, KNN, MLP, rede com embeddings (PyTorch) e o catálogo
scripts/
├── treinar_modelos.py   treina e salva os modelos em modelos_treinados/
└── gerar_producao.py    gera a amostra de produção (data/amostras/producao_100.csv)
```
