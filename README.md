# UC Aprendizagem Profunda — Trabalho Prático Módulo 1

## Grupo MEI-09

### Elementos do Grupo

| Nome | Nº de Aluno |
| - | - |
| Afonso Sousa | PG61506 |
| Daniel Lobo | PG60241 |
| Gabriel Ribeiro | PG60258 |
| Lingyun Zhu | PG57885 |
| Tomás Barbosa | PG60311 |

## Organização do Repositório

```text
.
├── Apresentação/                     # Vídeo da Apresentação 
│
├── data/                             # Datasets para treinamento e validação
│   ├── dataset-training.csv          # Dataset de treino versão 1
│   ├── dataset-exemplos.csv          # Dataset com exemplos iniciais
│   ├── train.csv                     # Dataset de treino total
│   ├── test.csv                      # Dataset de teste geral
│   ├── subm1_test.csv                # Dataset de teste - Submissão 1
│   ├── subm2_test.csv                # Dataset de teste - Submissão 2
│   └── subm3.csv                     # Dataset de teste - Submissão 3
│
├── src/                              # Framework NumPy de raiz
│   ├── neuralnet.py                  # Classe NeuralNetwork
│   ├── layers.py                     # DenseLayer e DropoutLayer
│   ├── activation.py                 # ReLUActivation e SoftmaxActivation
│   ├── losses.py                     # CategoricalCrossEntropy
│   ├── optimizer.py                  # Optimizer com SGD + Momentum
│   └── metrics.py                    # accuracy_score
│
├── plots/                            # Alguns gráficos gerados pelos notebooks
│
├── Subm1/                            # Submissão 1
├── Subm2/                            # Submissão 2
├── Subm3/                            # Submissão 3
│
├── 1_Treino_Numpy.ipynb              # Notebook de treino DNN NumPy (Subm1 e 2)
├── 2_Treino_PyTorch.ipynb            # Notebook de treino GRU PyTorch (Subm1 e 2)
├── 3_Submissao_Final.ipynb           # Notebook final de comparação de resultados
│
├── bert-llm-classifier.ipynb          # Classificador BERT e derivados
├── logistic_regression_base.ipynb     # Modelo base com Reg. logística
├── models_pytorch.ipynb               # Exploração de modelos PyTorch (Subm3)
├── transformer_classifier.ipynb       # Transformer feito de raiz
├── zero_few_shot.ipynb                # Abordagem zero/few-shot
├── zero_few_shot_rag.ipynb            # Zero/few-shot com RAG
├── zeroshot_gemini.ipynb              # Zero-shot com API Gemini
│
├── resultados.csv                     # Comparação de resultados (NB 3)
│
├── requirements.txt                   # Dependências
├── .gitignore                         # .gitignore
└── README.md                          # README
```

## Como Executar e Correr a Avaliação (Docente)

### Pré-requisitos

```bash
pip install -r requirements.txt
```

Os pesos e artefactos pré-treinados (`.pkl` e `.pth`) não se encontram na raiz do repositório por serem superiores a 100MB, para tal, executar o `1_Treino_Numpy.ipynb` (Run All) e o `2_Treino_PyTorch.ipynb` , pelo que **não é necessário correr os notebooks de treino novamente.** Para gerar as classificações num novo dataset de teste, basta executar os notebooks presentes na pasta **`Subm1/`**.

### 1. Avaliar o Modelo A (NumPy)

1. Abrir o ficheiro **`Subm1/subm1-g9-MEI-A.ipynb`**.
2. *(Opcional)* Se o nome ou caminho do ficheiro de validação for diferente, alterar a variável `PATH_DATASET_TESTE`. O formato esperado é CSV com separador `;` e colunas `ID;Text`.
3. Clicar em **"Run All"**.
4. O resultado será gerado e guardado como **`Subm1/subm1-g9-MEI-A.csv`**, contendo as colunas `ID;Text;Labels`.

### 2. Avaliar o Modelo B (PyTorch)

1. Abrir o ficheiro **`Subm1/subm1-g9-MEI-B.ipynb`**.
2. *(Opcional)* Se o nome ou caminho do ficheiro de validação for diferente, alterar a variável `PATH_DATASET_TESTE`.
3. Clicar em **"Run All"**.
4. O resultado será gerado e guardado como **`Subm1/subm1-g9-MEI-B.csv`**, contendo as colunas `ID;Text;Labels`.

> **Nota técnica:** Ambos os notebooks da pasta `Subm1` utilizam o `LabelEncoder` guardado em `modelo_numpy_artefactos.pkl` (na pasta raiz) para garantir o mapeamento rigoroso e uniforme das classes em ambos os modelos.
