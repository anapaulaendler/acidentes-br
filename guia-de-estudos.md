# Guia de estudos — Acidentes nas rodovias federais (Equipe 9)

Este guia explica o projeto para uma pessoa sem conhecimento de Ciência de Dados.
Leia as seções em ordem. No fim, há perguntas com respostas para treinar.

---

## 1. O projeto em poucas palavras

**A pergunta.** O que faz um acidente numa rodovia federal ser fatal?

**A hipótese.** A duplicação de trechos de pista simples reduz a chance de um acidente ser fatal.

**A decisão que o projeto apoia.** Se a hipótese for verdadeira, o projeto indica onde a duplicação tem o maior efeito.

**O que a equipe fez:**

1. Limpou os dados de acidentes da Polícia Rodoviária Federal (PRF).
2. Comparou os acidentes em pista simples e em pista dupla.
3. Treinou modelos de Machine Learning para medir o efeito da pista.
4. Fez uma aplicação web (em Flask) que mostra os resultados com gráficos.

---

## 2. Os dados

### 2.1 De onde vêm os dados

Os dados vêm dos [dados abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf).
A equipe usa o arquivo "agrupado por ocorrência": **cada linha é um acidente**.

- Período: de 2017 a 2025 (o ano de 2026 ainda não terminou, então fica fora).
- Total depois da limpeza: **632.693 acidentes**.
- Um arquivo por ano (`datatran2017.csv`, `datatran2018.csv` e assim por diante).

### 2.2 As colunas mais importantes

| Coluna | Significado |
|---|---|
| `data_inversa`, `horario`, `dia_semana` | Quando o acidente aconteceu |
| `uf`, `br`, `km`, `latitude`, `longitude` | Onde o acidente aconteceu |
| `tipo_pista` | Simples, dupla ou múltipla |
| `tipo_acidente` | Por exemplo: colisão frontal, colisão traseira, tombamento |
| `causa_acidente` | A causa que o agente da PRF anotou |
| `condicao_metereologica` | O clima (céu claro, chuva, neblina…) |
| `tracado_via` | Reta, curva, declive… |
| `classificacao_acidente` | Sem vítimas, com feridos ou **com vítimas fatais** |

### 2.3 O alvo

O **alvo** é o que os modelos tentam prever.
Neste projeto, o alvo é `fatal`: verdadeiro quando o acidente tem pelo menos uma morte.

- **6,9%** dos acidentes são fatais.
- Os acidentes fatais são uma **classe rara**. Esta característica muda a forma de avaliar os modelos (seção 5.4).

### 2.4 Colunas proibidas (vazamento)

Algumas colunas descrevem o resultado do acidente: `mortos`, `feridos`, `ilesos` e outras.
Se o modelo recebe a coluna `mortos`, ele "vê a resposta" antes de prever.
Este erro tem o nome de **vazamento do alvo** (*data leakage*). Os modelos nunca recebem estas colunas.

---

## 3. A limpeza dos dados

Dados reais têm erros. A limpeza corrige os erros antes da análise.

| Problema | Exemplo | Ação |
|---|---|---|
| Acidentes repetidos | 10 acidentes aparecem duas vezes | Remover as cópias |
| Valor faltante escrito como texto | 8.947 acidentes com clima "Ignorado" | Trocar por vazio (`NaN`) |
| Rodovia não registrada | 1.407 acidentes com `br` = 0 | Trocar por vazio |
| O mesmo traçado com textos diferentes | "Reta;Declive" e "Declive;Reta" | Pôr as partes em ordem alfabética (1.328 textos viram 384) |
| Coordenadas fora do Brasil | Longitude −405,9 | Trocar por vazio (54 acidentes) |

### 3.1 A mudança das causas em 2021

Em 2021, a PRF trocou a lista de causas. O dicionário de dados não informa esta mudança.

- Em 2017, a lista tinha 24 causas. A partir de 2021, tem cerca de 70.
- A maior causa de 2017 ("Falta de Atenção à Condução") não existe depois de 2020.

**Ação.** A equipe criou a coluna `grupo_causa`, com 10 grupos (por exemplo: "Velocidade", "Pedestre", "Veículo").
Cada grupo junta as causas antigas e as novas com o mesmo significado.

**Teste.** Se os grupos estão corretos, a proporção de cada grupo muda pouco de 2020 para 2021. O teste confirmou isso para 8 dos 10 grupos.

### 3.2 Os valores extremos (outliers)

Alguns acidentes têm muitas pessoas ou muitos veículos (o maior tem 131 veículos).
A regra do **IQR** marca 52.703 acidentes como outliers.

**Decisão: estes acidentes ficam na base.**
Nestes acidentes, 15,0% são fatais. Nos outros, 6,1%. Se a equipe remove estes acidentes, ela remove uma informação importante.

---

## 4. O que os dados crus mostram

"Dados crus" são os dados sem modelo, só com contagens.

| Fato | Número |
|---|---|
| Letalidade na pista simples | **9,3%** |
| Letalidade na pista dupla | **4,6%** |
| Letalidade na pista múltipla | **3,9%** |
| Anos em que a pista simples tem a letalidade mais alta | **9 de 9** |
| Colisões frontais entre os acidentes da pista simples | 11,7% |
| Colisões frontais entre os acidentes da pista dupla | 1,3% |
| Colisões frontais fatais na pista simples | 30,0% |

**Letalidade** é a porcentagem dos acidentes que são fatais.

**A explicação (o mecanismo).** Na pista simples, os dois sentidos não têm separação física.
Por isso, a colisão frontal é cerca de 9 vezes mais frequente, e ela é o tipo de acidente mais grave.

**Outros fatos:**

- O maior número de acidentes acontece às 18h. A maior letalidade acontece às 3h (12,0%).
- O grupo de causa mais letal é "Pedestre" (29,2%). O menos letal é "Veículo" (2,9%).
- Com neblina, a letalidade é 11,4%. Com chuva, é 5,8%, mais baixa que com céu claro (7,2%).

**Limite.** Os dados crus não provam a hipótese.
A pista simples pode ser mais letal por outros motivos, por exemplo porque fica em regiões rurais, com velocidade mais alta.
As próximas análises tentam separar esses motivos.

---

## 5. Os conceitos de Machine Learning (em linguagem simples)

### 5.1 O trecho

Um **trecho** é uma parte da rodovia: uma UF, uma BR e uma faixa de 5 km. Exemplo: BR-277, no Paraná, do km 630 ao km 635.

- A numeração do km recomeça em cada estado. Por isso, o trecho precisa da UF.
- A base tem 12.860 trechos. A análise usa só os **4.416 trechos com pelo menos 30 acidentes**.
- Motivo: num trecho com 3 acidentes, um único acidente fatal dá uma letalidade de 33%. O número não é confiável.

### 5.2 A regressão linear múltipla

A **regressão linear** encontra uma fórmula que prevê um número a partir de outros números.

- **O que a regressão prevê:** a letalidade de cada trecho.
- **As variáveis de entrada:** a porcentagem de pista simples, de curvas, de acidentes à noite, de chuva, de área urbana, de causas, o volume de acidentes e a região.
- **Múltipla** significa "com várias variáveis de entrada ao mesmo tempo".

**O coeficiente (β).** Cada variável recebe um número β.
O β mostra quanto a letalidade muda quando a variável aumenta e **as outras variáveis ficam iguais**.

**O resultado principal:**

> Com os outros fatores iguais, um trecho só com pista simples tem cerca de **4,0 pontos percentuais (pp)** a mais de letalidade que um trecho sem pista simples.

**Ponto percentual (pp)** é a diferença entre duas porcentagens. De 4% para 8% é um aumento de 4 pp.

### 5.3 Como avaliar a regressão

| Métrica | Significado | Resultado |
|---|---|---|
| **MAE** | O erro médio da previsão | 3,47 pp |
| **RMSE** | Também é um erro médio, mas os erros grandes pesam mais | 4,55 pp |
| **R²** | A fração da variação que o modelo explica (0 = nada, 1 = tudo) | 0,31 |

**A base de comparação.** Um modelo que sempre prevê a média tem MAE de 4,30 pp. A regressão é melhor que esta base.

**O R² de 0,31 é baixo.** A maior parte da variação vem de fatores que os dados da PRF não têm, por exemplo o número de veículos que passam no trecho.

### 5.4 A preparação dos dados para os modelos

- **One-Hot Encoding.** Um modelo só aceita números. O one-hot transforma uma categoria (por exemplo, a região "Sul") em colunas de 0 e 1.
- **Divisão treino/teste.** A equipe separa os dados em duas partes: 80% para o treino e 20% para o teste. O modelo aprende com o treino e é avaliado no teste, com dados que ele nunca viu.
- **Imputação sem vazamento.** A imputação preenche os valores vazios (por exemplo, com a mediana). A mediana é calculada **só no treino**. Se ela usar a base inteira, o treino recebe informação do teste.

### 5.5 A análise de sensibilidade

O resultado de 4,0 pp depende de duas escolhas: o mínimo de 30 acidentes por trecho e o peso de cada trecho.
A equipe testou outras escolhas:

- **Peso igual:** cada trecho conta uma vez. A pergunta é "quanto a pista simples muda a letalidade **num trecho típico**?".
- **Peso pelo número de acidentes:** um trecho com 3.000 acidentes conta como 100 trechos de 30. A pergunta é "quanto a pista simples muda a letalidade **para um acidente típico**?".

**Resultado.** O efeito é **positivo em todas as 8 combinações**, mas o valor varia de **0,8 a 5,2 pp**.
Por isso, a equipe apresenta o efeito como um intervalo.

### 5.6 O K-Means (agrupamento)

O **K-Means** é um modelo **não supervisionado**: ele não recebe a resposta.
Ele forma grupos de trechos parecidos.

- **As variáveis:** volume de acidentes, noite, chuva, curva, fim de semana, veículos e pessoas por acidente.
- **A pista e a letalidade ficam fora**, porque a análise compara a pista dentro de cada grupo.
- **Logaritmo.** Algumas variáveis têm valores muito grandes (um trecho tem 334 acidentes por ano). O logaritmo reduz esta diferença.
- **Padronização (Z-Score).** Cada variável fica com média 0 e desvio-padrão 1. Sem esta etapa, as variáveis com números grandes dominam o cálculo da distância.

**Como escolher o número de grupos (K):**

- **Método do cotovelo:** mostra a **inércia** (a distância entre os trechos e o centro do grupo) para cada K. A inércia sempre diminui quando K aumenta. O "cotovelo" é o ponto em que um grupo a mais ajuda pouco.
- **Silhueta:** de −1 a +1. Um valor alto significa grupos bem separados.

A equipe escolheu **K = 4**. A silhueta é **0,176** (baixa): os perfis dos trechos variam de forma contínua, sem fronteiras nítidas.

### 5.7 As personas

Uma **persona** é um nome de negócio para cada grupo.

| Persona | Característica | Letalidade média |
|---|---|---|
| Rurais com acidentes de vários veículos | 3,1 pessoas por acidente, 86% de pista simples | 10,1% |
| Noite e fim de semana | 42% dos acidentes à noite | 9,6% |
| Serra com curva e chuva | 53% dos acidentes com curva | 7,2% |
| Corredores urbanos movimentados | 34 acidentes por ano, 59% em área urbana | 4,6% |

**O resultado mais importante do projeto: a pista simples comparada com a dupla dentro de cada persona.**

| Persona | Pista dupla | Pista simples | Diferença |
|---|---|---|---|
| Serra com curva e chuva | 4,65% | 8,13% | +3,5 pp |
| Rurais com acidentes de vários veículos | 6,72% | 10,08% | +3,4 pp |
| Noite e fim de semana | 6,80% | 9,95% | +3,1 pp |
| Corredores urbanos movimentados | 3,80% | 3,90% | **+0,1 pp** |

**Conclusão.** Nos corredores urbanos, a duplicação quase não muda a letalidade.
A duplicação deve ter prioridade nos trechos de pista simples **fora** dos corredores urbanos.

### 5.8 Os classificadores

Um **classificador** prevê uma categoria. Aqui, ele prevê se um acidente vai ser fatal ou não.

| Modelo | Recall | Precisão | Acurácia |
|---|---|---|---|
| Regressão logística (com pesos de classe) | 71,9% | 20,0% | 78,3% |
| MLP do scikit-learn (sem pesos de classe) | 7,8% | 53,8% | 93,2% |

- **Recall:** dos acidentes fatais reais, quantos o modelo encontra.
- **Precisão:** dos alertas do modelo, quantos são fatais de verdade.
- **Acurácia:** quantas previsões estão corretas no total.

**Por que a acurácia engana.** Só 6,9% dos acidentes são fatais.
Um modelo que sempre diz "não fatal" acerta 93,1% dos casos, mas não encontra nenhum acidente fatal.
O MLP tem acurácia alta (93,2%), mas encontra só 7,8% dos fatais.

**Pesos de classe.** A regressão logística dá mais peso aos acidentes fatais durante o treino. Por isso, ela encontra mais acidentes fatais.

**A razão de chances (odds ratio).** Na regressão logística, a razão de chances da pista simples contra a pista dupla é **1,27**.
Com as outras variáveis iguais, a chance de um acidente ser fatal é cerca de 27% maior na pista simples.
O valor é menor que nos dados crus, porque o modelo também recebe o tipo de acidente (e a colisão frontal é o mecanismo da pista simples).

---

## 6. Os três níveis de evidência

| Nível | O que compara | Estado | Resultado |
|---|---|---|---|
| 1. Dados crus | Letalidade por tipo de pista | Feito | 9,3% contra 4,6% |
| 2. Comparação justa | Regressão por trecho e comparação dentro das personas | Feito | +0,8 a +5,2 pp; +3 pp fora dos corredores urbanos |
| 3. Experimento natural | O mesmo trecho antes e depois da duplicação | A fazer | — |

Cada nível controla mais fatores que o anterior. O nível 3 é o mais forte, porque compara o mesmo trecho com ele mesmo.

---

## 7. A aplicação web

### 7.1 Como rodar

```bash
pip install -r requirements.txt
python3 rodar.py
```

Abra http://127.0.0.1:5000. Na tela **Dados**, carregue uma base.

### 7.2 As telas

| Tela | Conteúdo |
|---|---|
| A hipótese | Os dados crus da seção 4 e os três níveis de evidência |
| Quando? | A letalidade por hora, dia da semana e mês |
| Onde? | O mapa dos acidentes fatais, as personas e a comparação dentro das personas |
| Condições | A letalidade por causa, clima e traçado, e a regressão por trecho |
| Modelos | Um cartão por modelo: a pergunta, os dados, o resultado e o limite |
| Dados | O upload, o relatório da limpeza e a tabela dos acidentes |

Cada gráfico fica num **cartão** com quatro partes: a afirmação, o gráfico, a leitura e o limite.

### 7.3 O que a aplicação faz

- **Aceita três formatos de arquivo:** CSV, JSON e XML.
- **Recalcula tudo com uma base nova.** Se o usuário envia outro ano ou outro estado com as mesmas colunas, as telas e os modelos mudam (o "re-treino").
- **Tem filtros:** UF, ano e tipo de pista. Os filtros valem para todas as telas.
- **Guarda os resultados num cache.** O cache é um dicionário. A chave tem a versão da base e os filtros. Uma base nova apaga o cache.

### 7.4 A organização do código

| Pasta | Conteúdo |
|---|---|
| `src/` | A limpeza, as análises e os modelos (usados pela aplicação e pelo notebook) |
| `app/` | A aplicação Flask: as telas, os templates HTML e os gráficos (Plotly) |
| `notebooks/analise_prf.ipynb` | A análise completa, passo a passo |
| `tests/` | Os testes automáticos (`python3 -m pytest`) |

---

## 8. As limitações

- **Associação não é causa.** As rodovias duplicadas também podem ter melhor sinalização e iluminação. Os dados não separam estes efeitos.
- **Os dados não têm o fluxo de veículos.** A PRF registra os acidentes, mas não o número de veículos que passam. Sem este número, não é possível calcular o risco por veículo.
- **A causa é uma opinião.** O agente da PRF anota a causa no local do acidente.
- **A lista de causas mudou em 2021.** O `grupo_causa` reduz o problema, mas o grupo "Via" ainda tem uma quebra.
- **Só rodovias federais.** Os resultados não valem para rodovias estaduais nem para ruas.

---

## 9. Perguntas para treinar

### 9.1 Perguntas simples

**1. Qual é a pergunta do projeto?**
O que faz um acidente numa rodovia federal ser fatal?

**2. Qual é a hipótese?**
A duplicação de trechos de pista simples reduz a chance de um acidente ser fatal.

**3. De onde vêm os dados?**
Dos dados abertos da Polícia Rodoviária Federal, de 2017 a 2025. Cada linha é um acidente.

**4. Quantos acidentes a base tem?**
632.693 acidentes depois da limpeza.

**5. O que é letalidade?**
A porcentagem dos acidentes que têm pelo menos uma morte.

**6. Qual é a letalidade na pista simples e na pista dupla?**
9,3% na pista simples e 4,6% na pista dupla.

**7. Por que a pista simples é mais perigosa?**
Os dois sentidos não têm separação física. A colisão frontal é cerca de 9 vezes mais frequente, e ela é o tipo de acidente mais grave.

**8. O que é um trecho?**
Uma parte da rodovia: uma UF, uma BR e uma faixa de 5 km.

**9. Em que formatos a aplicação aceita arquivos?**
CSV, JSON e XML.

**10. Quais são as telas da aplicação?**
A hipótese, Quando?, Onde?, Condições, Modelos e Dados.

**11. Qual hora tem a maior letalidade?**
As 3h da madrugada (12,0% dos acidentes são fatais).

**12. O que é o alvo?**
O que o modelo tenta prever. Aqui, o alvo é `fatal`: se o acidente teve pelo menos uma morte.

### 9.2 Perguntas médias

**13. Por que o modelo não recebe a coluna `mortos`?**
Porque ela descreve o resultado. O modelo "veria a resposta" antes de prever. Este erro tem o nome de vazamento do alvo.

**14. O que a limpeza fez com o clima "Ignorado"?**
Trocou o texto por um valor vazio. Para o pandas, "Ignorado" é uma categoria válida. Na verdade, significa que o clima não foi registrado.

**15. Por que a equipe criou a coluna `grupo_causa`?**
Porque a PRF trocou a lista de causas em 2021. Uma causa antiga pode não existir no ano seguinte. Os grupos juntam as causas antigas e as novas com o mesmo significado.

**16. Por que os outliers ficaram na base?**
Porque os acidentes com muitas pessoas têm uma letalidade de 15,0%, contra 6,1% nos outros. Se a equipe remove estes acidentes, ela remove uma informação importante para o modelo.

**17. Por que a análise usa só trechos com pelo menos 30 acidentes?**
Num trecho com poucos acidentes, a letalidade varia muito por acaso. Com 3 acidentes, um único acidente fatal dá 33%.

**18. O que significa "+4,0 pp" na regressão?**
Com os outros fatores iguais, um trecho só com pista simples tem 4 pontos percentuais a mais de letalidade que um trecho sem pista simples.

**19. O que é o R²? O R² de 0,31 é bom?**
O R² mostra a fração da variação que o modelo explica. O valor 0,31 é baixo: a maior parte da variação vem de fatores que os dados não têm, como o fluxo de veículos. Mesmo assim, o modelo é melhor que a média (MAE de 3,47 contra 4,30 pp).

**20. Por que a imputação usa só os dados de treino?**
Para não vazar informação do teste para o treino. Se a mediana usa a base inteira, o modelo recebe informação dos dados que ele deve prever.

**21. Para que serve o One-Hot Encoding?**
O modelo só aceita números. O one-hot transforma cada categoria (por exemplo, uma região) numa coluna de 0 e 1.

**22. Por que o K-Means não recebe a pista nem a letalidade?**
Porque a análise compara a pista simples e a dupla **dentro** de cada grupo. Se a pista entra no K-Means, os grupos separam as pistas, e a comparação fica impossível. A letalidade é o resultado que a análise mede.

**23. Por que padronizar os dados antes do K-Means?**
O K-Means mede distâncias. Sem padronização, uma variável com números grandes (por exemplo, o volume de acidentes) domina a distância, e as outras variáveis quase não contam.

**24. Por que K = 4?**
Depois de K = 4, cada grupo a mais reduz pouco a inércia (método do cotovelo). A silhueta de K = 4 é quase igual à de K = 3, e quatro grupos dão personas mais úteis para a decisão.

**25. Qual é o resultado mais importante do projeto?**
Dentro de cada persona, a pista simples tem cerca de 3 pp a mais de letalidade, exceto nos corredores urbanos (0,1 pp). A duplicação deve ter prioridade fora dos corredores urbanos.

**26. Por que a acurácia engana neste projeto?**
Só 6,9% dos acidentes são fatais. Um modelo que sempre diz "não fatal" acerta 93,1% dos casos, mas não encontra nenhum acidente fatal. Por isso, a equipe usa o recall e a precisão.

**27. Qual é a diferença entre recall e precisão?**
O recall mostra quantos acidentes fatais o modelo encontra. A precisão mostra quantos alertas do modelo são fatais de verdade. A regressão logística encontra 71,9% dos fatais, mas só 20,0% dos alertas dela são fatais.

**28. O que é a razão de chances (odds ratio) de 1,27?**
Com as outras variáveis iguais, a chance de um acidente ser fatal é cerca de 27% maior na pista simples que na pista dupla.

**29. O que é a análise de sensibilidade?**
Um teste do resultado com outras escolhas de método (o mínimo de acidentes e o peso dos trechos). O efeito da pista simples é positivo em todas as combinações, entre 0,8 e 5,2 pp.

**30. O projeto prova que a duplicação salva vidas?**
Não. O projeto mostra uma **associação** forte e consistente, mas não uma causa. As rodovias duplicadas podem ter outras diferenças. O nível 3 (o mesmo trecho antes e depois da duplicação) é o teste mais forte, e ainda está por fazer.

**31. O que acontece quando o usuário envia uma base nova?**
A aplicação lê o arquivo, faz a limpeza, apaga o cache e recalcula todas as telas e todos os modelos. Se o arquivo tem um erro (por exemplo, uma coluna faltando), a aplicação mostra a mensagem e mantém a base anterior.

**32. Por que o mapa não mostra todos os acidentes fatais?**
A base tem mais de 43 mil acidentes fatais. O mapa mostra uma amostra aleatória de 15 mil, para o navegador não ficar lento. O cartão do mapa informa isso.
