# Acidentes nas rodovias federais — Equipe 9

Análise de Data Science dos acidentes nas rodovias federais, feita num notebook, que testa uma pergunta:
**um modelo consegue prever se um acidente vai ser fatal, só com as condições do acidente?**

Fonte dos dados: [dados abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf), agrupados **por ocorrência** (`datatran<ano>.csv`).

## Como rodar

1. Instale as dependências (Python 3.11 ou mais recente):

   ```bash
   pip install -r requirements.txt
   ```

2. Coloque os arquivos `datatran2017.csv` a `datatran2026.csv` em `data/raw/`.

3. Abra o notebook `notebooks/analise_prf.ipynb` e rode as células em ordem.

4. Para treinar os modelos de novo, rode da raiz do projeto (a busca de hiperparâmetros primeiro, porque o treino usa o resultado dela):

   ```bash
   python3 -m scripts.buscar_hiperparametros                         # a busca, numa amostra do treino
   python3 -m scripts.treinar_modelos                                # todos os modelos
   python3 -m scripts.treinar_modelos "Rede com one-hot (PyTorch)"   # só esses
   ```

   Os modelos ficam salvos em `modelos_treinados/modelos.joblib`. O notebook só carrega esse arquivo, porque o treino com a base completa pode levar horas.

## Como funciona

### A pergunta e o alvo

Cada linha da base é um acidente. O alvo, `fatal`, é verdadeiro quando a `classificacao_acidente` é "Com Vítimas Fatais"
(`src/carregar.py`). A pergunta é se as **condições do acidente** bastam para prever isso.

Por isso, as colunas que só existem **depois** do acidente ficam fora do modelo (`VAZAMENTO` em `src/colunas.py`):
a própria classificação, mortos, feridos leves e graves, ilesos e ignorados. Com elas, o modelo "colaria" a resposta.

O modelo usa:

- **categóricas:** UF, BR, dia da semana, causa, tipo de acidente, fase do dia, sentido da via, condição meteorológica, tipo de pista, traçado e uso do solo;
- **numéricas:** km, pessoas e veículos;
- **a persona do trecho:** o K-Means da análise agrupa os trechos (UF, BR e faixa de 5 km) em 4 personas, e cada acidente recebe a persona do trecho dele.
  O passo `persona` do pipeline (`MarcarPersona`, em `src/modelos/personas.py`) roda o K-Means só com os acidentes de treino e sem o alvo, então o teste não muda as personas.
  Um trecho com menos de 30 acidentes no treino fica "Sem persona".

### Limpeza (`src/limpeza.py`)

- remove linhas duplicadas e marca os valores faltantes;
- padroniza os textos e junta as grafias diferentes da mesma causa;
- cria grupos de causa e de tipo de acidente;
- descarta coordenadas fora do Brasil: o acidente continua na base, só sai do mapa.

### Codificação (`src/features/encoding.py`)

- **One-hot** (regressão logística, árvore, KNN, Random Forest, MLP e a rede com one-hot): cada categoria vira uma coluna de 0 ou 1. As numéricas recebem a mediana quando faltam e são padronizadas.
- **Ordinal** (Gradient Boosting): o `HistGradientBoostingClassifier` não aceita a matriz esparsa do one-hot. Cada categoria vira um número inteiro, e o modelo trata a coluna como categórica, sem ordem. Acima de 255 categorias, as mais raras viram uma só.
- **Embeddings** (a rede com embeddings): cada categoria vira um número inteiro, e a rede aprende um vetor para ela durante o treino. Categorias parecidas tendem a ficar com vetores próximos. O `src/modelos/projecao.py` projeta os vetores das causas em 2D por PCA.

### Os modelos e os hiperparâmetros

Todos os modelos estão em `src/modelos/`. Cada um se registra no catálogo (`catalogo.py`) com seus hiperparâmetros e limites.

| Modelo | Hiperparâmetros | O que cada um muda |
|---|---|---|
| Regressão logística | `C`, `class_weight` | C: o inverso da regularização (menor = coeficientes mais contidos). É o modelo mais simples e serve de base para os outros. |
| Árvore de decisão | `profundidade`, `minimo_folha`, `class_weight` | Profundidade: quantas perguntas a árvore faz em sequência (mais profunda = decora mais). Mínimo por folha: quantos acidentes cada folha precisa ter (maior = regras mais gerais). |
| KNN | `vizinhos`, `pesos` | Vizinhos: quantos acidentes parecidos votam. Pesos: `distance` dá mais voto aos mais próximos. |
| Random Forest | `arvores`, `profundidade`, `minimo_folha`, `class_weight` | *Bagging*: cada árvore treina com uma amostra sorteada dos acidentes, e as árvores votam juntas. Mais árvores = voto mais estável. |
| Gradient Boosting | `iteracoes`, `taxa_aprendizado`, `profundidade`, `class_weight` | *Boosting*: cada árvore nova corrige o erro das anteriores. A taxa de aprendizado é o peso de cada correção. Para sozinho quando a validação interna para de melhorar. |
| MLP (scikit-learn) | `camadas`, `neuronios`, `alpha`, `taxa_aprendizado` | O tamanho da rede, a regularização (alpha alto = rede mais "contida") e o tamanho do passo do treino. Para sozinho quando a validação interna para de melhorar. |
| Redes em PyTorch (embeddings e one-hot) | `epocas`, `tamanho_lote`, `taxa_aprendizado`, `camadas`, `neuronios`, `dropout`, `class_weight` | As duas redes são iguais e só mudam a codificação, para isolar o efeito dos embeddings. Dropout desliga neurônios ao acaso no treino, o que reduz o overfitting. |

**Busca de hiperparâmetros** (`scripts/buscar_hiperparametros.py` e `src/modelos/busca.py`): o `RandomizedSearchCV` sorteia de 8 a 20 combinações por modelo, numa amostra de 150 mil acidentes do treino, com 3 fatias, e escolhe pela PR-AUC.
A melhor combinação só substitui os padrões do catálogo se tiver PR-AUC maior. As redes em PyTorch ficam com os padrões, para a comparação embeddings × one-hot.

O `class_weight="balanced"` dá mais peso à classe fatal no treino. Os acidentes fatais são minoria, e sem esse peso o modelo aprende a dizer "não fatal" para quase tudo.

### Como os modelos são avaliados (`src/avaliacao.py`)

- **Treino e teste:** 80% dos acidentes para o treino e 20% para o teste, estratificado (`train_test_split(stratify=y)`). O teste só é usado uma vez, para medir o modelo final.
- **Validação cruzada estratificada, só no treino:** 5 fatias. Cada fatia mantém a mesma proporção de fatais. O modelo treina com as outras fatias e é medido na fatia que não viu. O resultado é a média ± o desvio entre as fatias.
- **Recall é a métrica principal:** é a parte dos acidentes fatais de verdade que o modelo encontra. Deixar passar um acidente fatal custa mais que um alarme falso.
- **As outras métricas:**
  - precisão: dos acidentes que o modelo chamou de fatais, quantos eram;
  - F1: o equilíbrio entre recall e precisão;
  - PR-AUC: a qualidade da probabilidade, sem depender do limiar;
  - ROC-AUC: a chance de o modelo dar uma probabilidade maior a um acidente fatal que a um não fatal;
  - acurácia: aparece, mas engana, porque um modelo que diz "não fatal" para tudo já acerta quase sempre.
- **Treino × teste:** o modelo final também é medido numa amostra de 50 mil acidentes do treino. Se o treino é muito melhor que o teste, o modelo decorou os acidentes (*overfitting*).
- **Limiar:** 0,5 na probabilidade de ser fatal.
- **Amostra de produção:** `data/amostras/producao_100.csv` nunca entra no treino. Ela simula dados novos, que os modelos não viram.

## Estrutura

```
notebooks/
└── analise_prf.ipynb    a análise: dados, limpeza, regressão, k-means e os modelos treinados
src/                  dados e modelos
├── base.py           a base completa e a amostra de produção
├── carregar.py       leitura de CSV, JSON e XML
├── validar.py        as colunas obrigatórias
├── limpeza.py        limpeza e grupos de causa
├── colunas.py        os grupos de colunas e o alvo (fatal)
├── trechos.py        os trechos de rodovia da regressão
├── avaliacao.py      a validação cruzada e as métricas
├── features/         a codificação das colunas
└── modelos/          regressão logística, árvore de decisão, KNN, Random Forest, Gradient Boosting, MLP, rede com embeddings (PyTorch) e o catálogo
scripts/
├── buscar_hiperparametros.py   busca os hiperparâmetros e salva em modelos_treinados/busca.joblib
├── treinar_modelos.py   treina e salva os modelos em modelos_treinados/
└── gerar_producao.py    gera a amostra de produção (data/amostras/producao_100.csv)
```
