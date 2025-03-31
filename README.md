# Aplicação de Modelos de Regressão e Classificação com Validação via Simulações Monte Carlo

**Autor:** Samuel Lucas de Araujo Farias  
**Curso:** Ciência da Computação  
**Universidade:** Universidade de Fortaleza  

---

## Resumo

Este trabalho propõe a implementação e comparação de modelos preditivos utilizando o paradigma supervisionado, com aplicação em duas tarefas: regressão e classificação. Na etapa de regressão, o objetivo é prever o nível de atividade enzimática a partir das variáveis temperatura e pH da solução, empregando o método dos mínimos quadrados ordinários (MQO) tradicional, o MQO regularizado (Tikhonov) – testado para diferentes valores de λ (0, 0.25, 0.5, 0.75, 1) – e a predição pela média dos valores observáveis. A validação dos modelos é realizada por meio de simulações Monte Carlo, com 500 rodadas em que os dados são particionados em 80% para treinamento e 20% para teste, utilizando a soma dos desvios quadráticos (RSS) como métrica de desempenho.

Na etapa de classificação, o trabalho utiliza sinais eletromiográficos captados pelos sensores posicionados no Corrugador do Supercílio e no Zigomático Maior para identificar expressões faciais (neutro, sorriso, sobrancelhas levantadas, surpreso e rabugento). Os modelos implementados incluem o MQO tradicional, vários classificadores gaussianos (tradicional, com covariâncias iguais, com matriz agregada, regularizado via Friedman) e o classificador de Bayes Ingênuo. A performance dos modelos é avaliada via simulações Monte Carlo, também com 500 rodadas, adotando a acurácia (taxa de acerto) como critério de desempenho, seguida de uma análise estatística comparativa dos resultados.

---

## Palavras-Chave

MQO, regressão, classificadores gaussianos

---

## Sumário

- [Introdução](#i-introdução)
- [Metodologia](#ii-metodologia)
  - [Descrição e Organização dos Dados](#a-descrição-e-organização-dos-dados)
    - [Regressão](#a1-regressão)
    - [Classificação](#a2-classificação)
  - [Implementação dos Modelos](#b-implementação-dos-modelos)
    - [Modelos de Regressão](#b1-modelos-de-regressão)
    - [Modelos de Classificação](#b2-modelos-de-classificação)
  - [Validação via Simulações Monte Carlo](#c-validação-via-simulações-monte-carlo)
  - [Ambiente de Implementação e Ferramentas](#d-ambiente-de-implementação-e-ferramentas)
  - [Critérios de Avaliação e Apresentação dos Resultados](#e-critérios-de-avaliação-e-apresentação-dos-resultados)
- [Resultados](#iii-resultados)
  - [Regressão](#a-resultados-da-regressão)
  - [Classificação](#b-resultados-da-classificação)
- [Conclusão](#iv-conclusão)

---

## I. Introdução

O avanço tecnológico e o aumento na disponibilidade de dados têm impulsionado o uso de técnicas de aprendizado supervisionado em diversas áreas do conhecimento. Nesse contexto, a predição de variáveis e a classificação de padrões emergem como desafios relevantes tanto em problemas práticos quanto em estudos teóricos. 

Este trabalho concentra-se em duas tarefas fundamentais:  
- **Regressão:** Previsão de atividade enzimática com base em variáveis experimentais, como temperatura e pH, utilizando métodos como o MQO tradicional e suas versões regularizadas.  
- **Classificação:** Identificação de expressões faciais a partir de sinais eletromiográficos coletados por sensores posicionados em áreas específicas do rosto, exigindo abordagens que lidem com dados não linearmente separáveis.

A introdução contextualiza o problema, destacando a importância das técnicas de predição em cenários reais e os desafios envolvidos na modelagem e validação dos modelos propostos. Com uma metodologia fundamentada em simulações extensas, o estudo analisa comparativamente as técnicas para identificar seus pontos fortes e limitações.

---

## II. Metodologia

A metodologia adotada foi estruturada em diversas etapas, abrangendo o pré-processamento dos dados, a implementação dos modelos preditivos e a validação via simulações Monte Carlo.

### A. Descrição e Organização dos Dados

#### A.1. Regressão

- **Dados:**  
  O arquivo `atividade_enzimatica.csv` contém medições experimentais com três variáveis: temperatura, pH da solução e nível de atividade enzimática.
  
- **Organização:**  
  - As variáveis independentes (temperatura e pH) foram organizadas em uma matriz **X** de dimensão _N × 2_.
  - A variável dependente (atividade enzimática) foi armazenada em um vetor **y** de dimensão _N × 1_.
  
- **Visualização:**  
  Foi construído um gráfico de dispersão 3D para identificar padrões entre temperatura, pH e atividade enzimática.

  <div align="center">
    <img src="https://github.com/user-attachments/assets/b14d4695-c745-422b-ac1f-0d3360a98c1e" alt="Figura 1: Distribuição dos dados em gráfico 3D" style="display: block; margin-left: auto; margin-right: auto; width:300px;" />
  </div>

#### A.2. Classificação

- **Dados:**  
  O arquivo `EMGsDataset.csv` contém:
  - Leituras de sinais eletromiográficos de dois sensores (Corrugador do Supercílio e Zigomático Maior).
  - Uma terceira coluna com os rótulos de expressão facial (1 – Neutro, 2 – Sorriso, 3 – Sobrancelhas levantadas, 4 – Surpreso, 5 – Rabugento).

- **Organização:**  
  - Os sinais dos sensores foram organizados em uma matriz **X** com formato _2 × N_.
  - Os rótulos foram armazenados em um vetor **y**.
  - Para alguns modelos, os dados foram reorganizados (por exemplo, transpondo para formar **X = N × 2** ou convertendo rótulos para formato one-hot).

- **Visualização:**  
  Foi criado um gráfico 3D, utilizando cores distintas para cada classe, a fim de analisar a separabilidade dos dados.

  <div align="center">
    <img src="https://github.com/user-attachments/assets/c804a567-4fe7-46dc-bbcb-dcd3619580aa" alt="Figura 2: Distribuição dos dados dos sensores e expressões em gráfico 3D" style="display: block; margin-left: auto; margin-right: auto; width:300px;" />
  </div>

### B. Implementação dos Modelos

A implementação dos modelos foi realizada manualmente, sem o uso de bibliotecas prontas, conforme as restrições do trabalho.

#### B.1. Modelos de Regressão

1. **MQO Tradicional:**  
   - Foi incluído o intercepto por meio de uma coluna de 1s em **X**.
   - Os parâmetros β foram estimados pela fórmula:  
     
     $$ \beta = (X^T X)^{-1} X^T y $$

   - A predição é feita pela multiplicação de **X** por **β**.

2. **MQO Regularizado (Tikhonov):**  
   - Utiliza um termo de regularização com hiperparâmetro λ (testados: 0, 0.25, 0.5, 0.75, 1), sem penalizar o intercepto.
   - A fórmula para o cálculo é:  

     $$ \beta_{\text{reg}} = (X^T X + \lambda I')^{-1} X^T y $$  
     
     Onde **I′** é a matriz identidade modificada (com zero na primeira posição).

3. **Média dos Valores Observáveis:**  
   - Utiliza-se a média dos valores observados de **y** como predição constante.

#### B.2. Modelos de Classificação

1. **MQO para Classificação:**  
   - Similar ao modelo de regressão, porém com arredondamento dos valores preditos para corresponder aos rótulos de classe.

2. **Classificador Gaussiano Tradicional:**  
   - Calcula-se, para cada classe, a média e a matriz de covariância.
   - Utiliza-se a função densidade de probabilidade da distribuição normal multivariada, com regularização na matriz de covariância.

3. **Classificador Gaussiano com Covariâncias Iguais:**  
   - Utiliza-se uma única matriz de covariância calculada globalmente para todas as classes, mantendo as médias específicas.

4. **Classificador Gaussiano com Matriz Agregada:**  
   - Combina as covariâncias de cada classe para formar uma matriz pooled (agregada).

5. **Classificador Gaussiano Regularizado (Friedman):**  
   - Aplica regularização na matriz pooled testando os valores de λ (0, 0.25, 0.5, 0.75, 1).

6. **Classificador de Bayes Ingênuo:**  
   - Estima, para cada classe, a média e a variância de cada feature, assumindo independência condicional entre as características.
   - A classificação é realizada pela multiplicação dos PDFs univariados para cada feature, considerando os priors.

### C. Validação via Simulações Monte Carlo

Para avaliar a robustez dos modelos, foram realizadas 500 simulações (R = 500).

#### C.1. Particionamento dos Dados

- Os dados foram divididos aleatoriamente:
  - **80%** para treinamento.
  - **20%** para teste.

#### C.2. Cálculo das Métricas de Desempenho

- **Regressão:**  
  - Cálculo da soma dos desvios quadráticos (RSS) entre os valores reais e os preditos.
  - Estatísticas (média, desvio-padrão, máximo e mínimo) dos RSS foram calculadas.

- **Classificação:**  
  - Cálculo da acurácia (taxa de acerto) para cada rodada.
  - Estatísticas similares (média, desvio-padrão, máximo e mínimo) foram obtidas.

#### C.3. Armazenamento e Apresentação dos Resultados

- Os resultados de cada rodada foram armazenados em listas específicas para cada modelo.
- Posteriormente, as estatísticas foram organizadas em tabelas e complementadas com gráficos.

### D. Ambiente de Implementação e Ferramentas

#### D.1. Linguagem de Programação e Bibliotecas

- **Python:**  
  - Biblioteca NumPy para operações matemáticas e manipulação de arrays.
  - Matplotlib (incluindo o módulo mpl_toolkits) para visualização dos dados e resultados.

#### D.2. Restrições

- Não foram utilizadas bibliotecas com funções prontas para os modelos, exigindo a implementação manual dos algoritmos.

#### D.3. Ambiente Computacional

- As simulações e análises estatísticas foram conduzidas em ambiente local, com os códigos organizados para execução sequencial das etapas.

### E. Critérios de Avaliação e Apresentação dos Resultados

#### E.1. Critérios de Desempenho

- **Regressão:** Utilizou-se a soma dos desvios quadráticos (RSS).
- **Classificação:** Foi adotada a acurácia (taxa de acerto).

#### E.2. Análise Estatística

- Para cada modelo, foram calculadas medidas estatísticas (média, desvio-padrão, máximo e mínimo) a partir das 500 simulações.

#### E.3. Discussão

- A comparação dos modelos considera a sensibilidade aos hiperparâmetros (como os valores de λ na regularização) e a variabilidade dos dados, possibilitando identificar quais métodos apresentam melhor desempenho e consistência.

---

## III. Resultados

Nesta seção são apresentados os resultados obtidos para as tarefas de regressão e classificação, com base nas simulações Monte Carlo (R = 500) e na implementação manual dos modelos.

### A. Resultados da Regressão

#### A.1. Visualização e Ajuste dos Modelos

Foram utilizados dados do arquivo `atividade_enzimatica.csv`. A visualização em gráfico 3D permitiu identificar que, embora os dados não apresentem comportamento estritamente linear, é plausível aplicar métodos de regressão linear.

- **Modelos Implementados:**  
  1. **MQO Tradicional:**  
     - Estimativa de β pela expressão $$ (X^T X)^{-1} X^T y $$.
     
     <div align="center">
       <img src="https://github.com/user-attachments/assets/b9e65e79-9d72-4449-bf62-f3b6ffad2b0d" alt="Figura 3: Superfície de regressão (vermelho) ajustada pelo MQO tradicional com dados reais em azul" style="display: block; margin-left: auto; margin-right: auto; width:300px;" />
     </div>

  2. **MQO Regularizado (Tikhonov):**  
     - Teste dos valores de λ = {0, 0.25, 0.5, 0.75, 1}.
     - Fórmula:  
       
       $$ \beta_{\text{reg}} = (X^T X + \lambda I')^{-1} X^T y $$
  
  3. **Média dos Valores Observáveis:**  
     - Predição constante utilizando a média de **y**.
     
     <div align="center">
       <img src="https://github.com/user-attachments/assets/f08f946a-6a5b-4328-8d46-c6b22ca3c6f3" alt="Figura 4: Superfície de regressão constante ajustada pela média dos valores observáveis com dados reais em azul" style="display: block; margin-left: auto; margin-right: auto; width:300px;" />
     </div>

#### A.2. Validação via Monte Carlo e Análise dos RSS

- Realizou-se a validação com 500 rodadas, particionando 80% dos dados para treinamento e 20% para teste.
- Para cada modelo, foram calculadas as estatísticas do RSS (média, desvio-padrão, máximo e mínimo).

<div align="center">
  <img src="https://github.com/user-attachments/assets/e806a4ea-ebc3-400a-aa1e-471b3774bee4" alt="Tabela 1" style="display: block; margin-left: auto; margin-right: auto; width:300px;" />
</div>

#### A.3. Discussão dos Resultados de Regressão

- **Comparação entre Modelos:**  
  O modelo baseado na média apresentou um RSS médio muito mais elevado em comparação com os modelos baseados em MQO, demonstrando a eficácia da regressão linear.

- **Efeito de λ na Regularização:**  
  Para λ = 0 o resultado equivale ao MQO tradicional. À medida que λ aumenta, os valores de RSS sofrem variações mínimas, sugerindo pouca influência da regularização nos dados analisados.

- **Variabilidade:**  
  O desvio-padrão dos RSS permanece relativamente constante, indicando consistência dos resultados ao longo das simulações.

---

### B. Resultados da Classificação

#### B.1. Visualização Inicial e Organização dos Dados

Utilizou-se o arquivo `EMGsDataset.csv` para a classificação. A visualização dos dados em um gráfico 3D evidenciou que as classes não são linearmente separáveis.

<div align="center">
  <img src="https://github.com/user-attachments/assets/83604df6-0060-4b7c-8cc7-0944289e9b73" alt="Figura 5: Gráfico 3D com distribuição dos dados dos sensores e expressões" style="display: block; margin-left: auto; margin-right: auto; width:300px;" />
</div>

#### B.2. Modelos de Classificação Implementados

Os seguintes modelos foram implementados:

1. **MQO para Classificação:**  
   - Resultados com arredondamento dos valores preditos.

   <div align="center">
     <img src="https://github.com/user-attachments/assets/891002fe-5c9b-49ad-a8c6-7cae72f53ebb" alt="Figura 6: Resultados do MQO para classificação com dados reais (azul) e predições (vermelho)" style="display: block; margin-left: auto; margin-right: auto; width:300px;" />
   </div>

2. **Classificador Gaussiano Tradicional:**  
   
   <div align="center">
     <img src="https://github.com/user-attachments/assets/560db5cd-db0d-42bf-acfc-3eb057a697f0" alt="Figura 7: Resultados do Classificador Gaussiano Tradicional com dados reais (azul) e predições (vermelho)" style="display: block; margin-left: auto; margin-right: auto; width:300px;" />
   </div>

3. **Classificador Gaussiano com Covariâncias Iguais:**  
   
   <div align="center">
     <img src="https://github.com/user-attachments/assets/a577c5cf-1d5c-4db7-9e95-95efef92d014" alt="Figura 8: Resultados do Classificador Gaussiano com Covariâncias Iguais" style="display: block; margin-left: auto; margin-right: auto; width:300px;" />
   </div>

4. **Classificador Gaussiano com Matriz Agregada:**  
   
   <div align="center">
     <img src="https://github.com/user-attachments/assets/5c1f0186-ac84-42d9-a4c2-24309ce6966d" alt="Figura 9: Resultados do Classificador Gaussiano com Matriz Agregada" style="display: block; margin-left: auto; margin-right: auto; width:300px;" />
   </div>

5. **Classificador Gaussiano Regularizado (Friedman):**  
   - Teste de λ = {0, 0.25, 0.5, 0.75, 1}.

6. **Classificador de Bayes Ingênuo:**  
   
   <div align="center">
     <img src="https://github.com/user-attachments/assets/b76c4517-234b-4a30-a85c-b28c8256b256" alt="Figura 10: Resultados do Classificador de Bayes Ingênuo" style="display: block; margin-left: auto; margin-right: auto; width:300px;" />
   </div>

#### B.3. Validação via Monte Carlo e Acurácias

- Foram realizadas 500 simulações, utilizando 80% dos dados para treinamento e 20% para teste.
- Para cada modelo, foram calculadas as estatísticas das acurácias (média, desvio-padrão, máximo e mínimo).

<div align="center">
  <img src="https://github.com/user-attachments/assets/59420994-9ce1-4509-96f4-f8a88feb8e9a" alt="Tabela 2" style="display: block; margin-left: auto; margin-right: auto; width:300px;" />
</div>

#### B.4. Discussão dos Resultados de Classificação

- **Desempenho Geral:**  
  O modelo MQO para classificação atingiu uma acurácia em torno de 52,86%, bem inferior aos modelos probabilísticos, que alcançaram valores próximos ou acima de 96%.

- **Comparação entre Modelos Gaussianos:**  
  Os diferentes modelos gaussianos apresentaram acurácias elevadas, com pequenas variações entre si. O uso de covariâncias iguais mostrou uma ligeira redução em relação aos demais.

- **Efeito da Regularização:**  
  A regularização (Friedman) demonstrou uma leve diminuição na acurácia à medida que λ aumenta, mas sem impacto expressivo.

---

## IV. Conclusão

Este trabalho demonstrou a eficácia das técnicas de regressão e classificação na predição de atividade enzimática e no reconhecimento de expressões faciais. 

- **Regressão:**  
  Os métodos baseados em MQO (tradicional e regularizado) apresentaram desempenho consistente e superior à predição pela média, evidenciando a capacidade da abordagem linear em capturar relações entre variáveis experimentais.

- **Classificação:**  
  Os classificadores baseados em modelos gaussianos e o classificador de Bayes Ingênuo alcançaram acurácias significativamente maiores que o MQO adaptado para classificação, ressaltando a importância das abordagens probabilísticas em problemas de dados não linearmente separáveis.

A validação por meio de simulações Monte Carlo (500 rodadas) reforçou a robustez dos resultados, permitindo uma análise estatística detalhada dos modelos e evidenciando a influência de hiperparâmetros como os valores de λ. Este estudo fornece diretrizes valiosas para a implementação e avaliação de modelos preditivos e abre caminho para investigações futuras na área de aprendizado supervisionado.
