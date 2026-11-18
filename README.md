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

4. Para treinar os modelos de novo (por exemplo, depois de adicionar dados), há duas opções:
   - **Na app**, tela **Modelos**, cartão **Re-treinar um modelo**: escolha o modelo, ajuste os hiperparâmetros
     (profundidade da árvore, vizinhos do KNN, camadas e neurônios das redes…) e clique em **Re-treinar**.
     O modelo treina com a base atual e é salvo; os outros continuam como estão.
   - **Pelo terminal**, todos os modelos com os hiperparâmetros padrão (depois, reinicie a app):

     ```bash
     python3 -m scripts.treinar_modelos
     ```

O mapa precisa de internet (fundo do OpenStreetMap). O resto da app funciona sem internet.

## Telas

| Tela | O que mostra |
|---|---|
| Acidentes | O mapa dos acidentes, paginado, os acidentes por UF (fatais e não fatais), os fatais por fase do dia (pizza) e a letalidade por tipo de acidente |
| Modelos | O desempenho dos classificadores na validação cruzada, a rede com embeddings, a previsão dos dados adicionados e o re-treino com hiperparâmetros escolhidos |
| Dados | A lista paginada de todos os acidentes da base |
| Dados adicionados | O envio de arquivos e a lista paginada só com os acidentes enviados |

Cada resultado fica num **cartão de insight**: a afirmação, a evidência, a leitura e o limite.
Um arquivo enviado entra na base e muda as telas na hora. Os modelos só mudam com um novo treino.
Com dados novos, a tela Modelos oferece duas saídas: **prever** (leve: os modelos ficam como estão e dizem onde os dados novos se encaixam)
ou **re-treinar** (pesado: validação cruzada e treino com a base toda, que na base completa pode levar horas).

Os filtros da barra lateral (acidentes fatais ou não, UF e ano) valem para o mapa, o gráfico por UF e as listas.

## Como funciona

### A pergunta e o alvo

Cada linha da base é um acidente. O alvo, `fatal`, é verdadeiro quando a `classificacao_acidente` é "Com Vítimas Fatais"
(`src/carregar.py`). A pergunta é se as **condições do acidente** bastam para prever isso.

Por isso, as colunas que só existem **depois** do acidente ficam fora do modelo (`VAZAMENTO` em `src/colunas.py`):
a própria classificação, mortos, feridos leves e graves, ilesos e ignorados. Com elas, o modelo "colaria" a resposta.

O modelo usa:

- **categóricas:** UF, BR, dia da semana, causa, tipo de acidente, fase do dia, sentido da via, condição meteorológica, tipo de pista, traçado e uso do solo;
- **numéricas:** km, pessoas e veículos.

### Limpeza (`src/limpeza.py`)

- remove linhas duplicadas e marca os valores faltantes;
- padroniza os textos e junta as grafias diferentes da mesma causa;
- cria grupos de causa e de tipo de acidente;
- descarta coordenadas fora do Brasil: o acidente continua na base, só sai do mapa.

### Codificação (`src/features/encoding.py`)

- **One-hot** (árvore, KNN, MLP e a rede com one-hot): cada categoria vira uma coluna de 0 ou 1. As numéricas recebem a mediana quando faltam e são padronizadas.
- **Embeddings** (a rede com embeddings): cada categoria vira um número inteiro, e a rede aprende um vetor para ela durante o treino. Categorias parecidas tendem a ficar com vetores próximos. A tela Modelos mostra isso com os vetores das causas projetados em 2D por PCA.

### Os modelos e os hiperparâmetros

Todos os modelos estão em `src/modelos/`. Cada um se registra no catálogo (`catalogo.py`) com seus hiperparâmetros e limites. A tela de re-treino monta os campos a partir desse catálogo.

| Modelo | Hiperparâmetros | O que cada um muda |
|---|---|---|
| Árvore de decisão | `profundidade`, `minimo_folha`, `class_weight` | Profundidade: quantas perguntas a árvore faz em sequência (mais profunda = decora mais). Mínimo por folha: quantos acidentes cada folha precisa ter (maior = regras mais gerais). |
| KNN | `vizinhos`, `pesos` | Vizinhos: quantos acidentes parecidos votam. Pesos: `distance` dá mais voto aos mais próximos. |
| MLP (scikit-learn) | `camadas`, `neuronios`, `alpha`, `taxa_aprendizado` | O tamanho da rede, a regularização (alpha alto = rede mais "contida") e o tamanho do passo do treino. Para sozinho quando a validação interna para de melhorar. |
| Redes em PyTorch (embeddings e one-hot) | `epocas`, `tamanho_lote`, `taxa_aprendizado`, `camadas`, `neuronios`, `dropout`, `class_weight` | As duas redes são iguais e só mudam a codificação, para isolar o efeito dos embeddings. Dropout desliga neurônios ao acaso no treino, o que reduz o overfitting. |

O `class_weight="balanced"` dá mais peso à classe fatal no treino. Os acidentes fatais são minoria, e sem esse peso o modelo aprende a dizer "não fatal" para quase tudo.

### Como os modelos são avaliados (`src/avaliacao.py`)

- **Validação cruzada estratificada:** 5 fatias por padrão, e 3 ou 5 no re-treino pela tela. Cada fatia mantém a mesma proporção de fatais. O modelo treina com as outras fatias e é medido na fatia que não viu. A tela mostra a média ± o desvio entre as fatias.
- **Recall é a métrica principal:** é a parte dos acidentes fatais de verdade que o modelo encontra. Deixar passar um acidente fatal custa mais que um alarme falso.
- **As outras métricas:**
  - precisão: dos acidentes que o modelo chamou de fatais, quantos eram;
  - F1: o equilíbrio entre recall e precisão;
  - PR-AUC: a qualidade da probabilidade, sem depender do limiar;
  - acurácia: aparece, mas engana, porque um modelo que diz "não fatal" para tudo já acerta quase sempre.
- **Limiar:** 0,5 na probabilidade de ser fatal.
- **Amostra de produção:** `data/amostras/producao_100.csv` nunca entra no treino. Ela simula dados novos, que os modelos não viram, para testar o botão **Prever**.

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
