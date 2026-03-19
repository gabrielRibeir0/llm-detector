# UC Aprendizagem Profunda — Trabalho Prático Módulo 1

## Grupo MEI-09

### Elementos do Grupo

| Nome | Nº de Aluno |
|------|-------------|
| Afonso Sousa | PG61506 |
| Daniel Lobo | PG60241 |
| Gabriel Ribeiro | PG60258 |
| Tomás Barbosa | PG60311 |

---

## Objetivo

Desenvolvimento de modelos de Machine/Deep Learning para classificar textos como gerados por diferentes modelos de Inteligência Artificial ou escritos por humanos. Problema de classificação multi-classe com as seguintes classes:

| Classe | Descrição |
|--------|-----------|
| `Human` | Texto escrito por seres humanos |
| `Google` | Texto gerado por modelos Gemma/Gemini |
| `Meta` | Texto gerado por modelos LLaMA |
| `OpenAI` | Texto gerado por modelos GPT |
| `Mistral` | Texto gerado por modelos da família Mistral |

> Nota: no dataset atualmente incluído no repositório, não existem ainda exemplos rotulados como `Mistral`.
> Enquanto esses dados não forem recolhidos, os notebooks assinalam esta lacuna e usam `Anthropic` como substituto temporário para manter o treino multi-classe estável.

---

## Organização do Repositório

```text
.
├── data/
│   ├── dataset-training.csv       # Dataset de treino principal
│   └── subm1.csv                  # Dataset de validação fornecido pelo docente (formato: ID;Text)
│
├── src/                           # Framework NumPy implementado de raiz (Tarefa 2)
│   ├── neuralnet.py               # Classe principal NeuralNetwork (loop de treino, forward/backward)
│   ├── layers.py                  # DenseLayer e DropoutLayer
│   ├── activation.py              # ReLUActivation e SoftmaxActivation
│   ├── losses.py                  # CategoricalCrossEntropy (loss + derivada)
│   ├── optimizer.py               # Optimizer com SGD + Momentum
│   └── metrics.py                 # accuracy_score
│
├── plots/                         # Gráficos gerados automaticamente pelos notebooks de treino
│
├── 1_Treino_Numpy.ipynb           # Notebook de treino da DNN NumPy (Tarefa 2)
├── 2_Treino_PyTorch.ipynb         # Notebook de treino do GRU PyTorch (Tarefa 3)
│
├── modelo_numpy_artefactos.pkl    # Modelo NumPy treinado + TF-IDF vectorizer + LabelEncoder
├── modelo_pytorch_gru.pth         # Pesos do modelo PyTorch GRU (melhor época por val_loss)
├── pytorch_vocab.pkl              # Vocabulário construído para o modelo PyTorch
│
└── Subm1/                         # PASTA DE SUBMISSÃO OFICIAL
    ├── subm1-g9-MEI-A.ipynb       # Notebook de Inferência - Modelo A (NumPy)
    ├── subm1-g9-MEI-A.csv         # Resultados gerados pelo Modelo A
    ├── subm1-g9-MEI-B.ipynb       # Notebook de Inferência - Modelo B (PyTorch)
    └── subm1-g9-MEI-B.csv         # Resultados gerados pelo Modelo B
```

---

## Modelos Desenvolvidos

### Modelo A — NumPy DNN + Baseline Logístico (Tarefa 2, implementação própria)

Deep Neural Network e baseline de Regressão Logística implementados **de raiz em NumPy**, sem uso de bibliotecas de ML/DL externas no pipeline da Tarefa 2.

**Pipeline:**
1. Limpeza de texto (lowercase, remoção de HTML e caracteres especiais)
2. Extração de features com **TF-IDF próprio** (até 5000 features)
3. Divisão estratificada **train/val/test (70/15/15)**
4. Treino da DNN com backpropagation manual, Dropout, momentum e early stopping
5. Treino adicional de baseline logístico em NumPy para comparação

**Arquitetura:**
```
Input (TF-IDF, ~5000 features)
  → Dense(256) → ReLU → Dropout(0.4)
  → Dense(128) → ReLU → Dropout(0.3)
  → Dense(64)  → ReLU → Dropout(0.2)
  → Dense(5)   → Softmax
```

**Hiperparâmetros:** `epochs=50`, `batch_size=16`, `lr=0.01`, `momentum=0.9`

**Ficheiro gerado (já na raiz):** `modelo_numpy_artefactos.pkl`

---

### Modelo B — PyTorch GRU Bidirecional (Tarefa 3)

GRU bidirecional com embeddings aprendidos, implementado em **PyTorch**.

**Pipeline:**
1. Limpeza e tokenização do texto
2. Construção de vocabulário (top 5000 palavras)
3. Encoding de sequências com padding (max_len=100)
4. Divisão treino/validação/teste (70/15/15)
5. Treino com validação por época, `ReduceLROnPlateau` e gradient clipping

**Arquitetura:**
```
Input (token IDs, seq_len=100)
  → Embedding(vocab=5000, dim=128)
  → GRU Bidirecional (hidden=128, 2 layers, dropout=0.3)
  → Concatenação das hidden states [forward + backward]
  → Dropout(0.4)
  → Linear(256 → 5)
```

**Hiperparâmetros:** `epochs=30`, `batch_size=16`, `lr=0.001`, `weight_decay=1e-4`

**Ficheiros gerados (já na raiz):** `modelo_pytorch_gru.pth` + `pytorch_vocab.pkl`

---

## Como Executar e Correr a Avaliação (Docente)

### Pré-requisitos
```bash
pip install numpy pandas scikit-learn matplotlib torch
```

### Construção dos Datasets (Tarefa 1)

O projeto usa datasets em inglês na pasta `data/`, com estrutura tabular (`ID;Text;Label`):

- `data/dataset-training.csv`
- `data/dataset-training-final.csv`
- `data/dataset-exemplos.csv`
- `data/subm1.csv` (validação externa para submissão)

Recomendação para fechar integralmente a Tarefa 1: documentar no relatório final as fontes originais (Kaggle/Hugging Face/APIs) e respetivo processo de limpeza/normalização para cada classe.

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
