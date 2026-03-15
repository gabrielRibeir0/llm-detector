# UC Aprendizagem Profunda — Trabalho Prático Módulo 1

## Grupo 14

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
| `Mistral` | Texto gerado por Mistral-7B ou similares |

---

## Organização do Repositório

```
.
├── data/
│   ├── dataset-exemplos.csv       # Dataset de treino principal (formato: ID;Text;Label)
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
├── plots/                         # Gráficos gerados automaticamente pelos notebooks
│
├── 1_Treino_Numpy.ipynb           # Treino da DNN NumPy (Tarefa 2)
├── 2_Treino_PyTorch.ipynb         # Treino do GRU PyTorch (Tarefa 3)
├── 3_Submissao_Final.ipynb        # Inferência + comparação dos modelos (Avaliação)
│
├── modelo_numpy_artefactos.pkl    # Modelo NumPy treinado + TF-IDF vectorizer + LabelEncoder
├── modelo_pytorch_gru.pth         # Pesos do modelo PyTorch GRU (melhor época por val_loss)
├── pytorch_vocab.pkl              # Vocabulário construído para o modelo PyTorch
└── pytorch_metrics.pkl            # Métricas de avaliação do modelo PyTorch
```

---

## Modelos Desenvolvidos

### Modelo 1 — NumPy DNN (Tarefa 2, implementação própria)

Deep Neural Network implementada **de raiz em NumPy**, sem uso de bibliotecas de ML externas.

**Pipeline:**
1. Limpeza de texto (lowercase, remoção de HTML e caracteres especiais)
2. Extração de features com **TF-IDF** (até 5000 features, stop words inglesas removidas)
3. Divisão treino/teste estratificada (80/20)
4. Treino da DNN com backpropagation manual

**Arquitetura:**
```
Input (TF-IDF, ~2500 features)
  → Dense(256) → ReLU → Dropout(0.4)
  → Dense(128) → ReLU → Dropout(0.3)
  → Dense(64)  → ReLU → Dropout(0.2)
  → Dense(5)   → Softmax
```

**Hiperparâmetros:** `epochs=50`, `batch_size=16`, `lr=0.01`, `momentum=0.9`

**Ficheiro gerado:** `modelo_numpy_artefactos.pkl`

---

### Modelo 2 — PyTorch GRU Bidirecional (Tarefa 3)

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

**Ficheiro gerado:** `modelo_pytorch_gru.pth` + `pytorch_vocab.pkl`

---

## Métricas de Avaliação

Ambos os modelos são avaliados com as seguintes métricas no conjunto de teste:

| Métrica | Descrição |
|---------|-----------|
| **Test Loss** | Cross-Entropy no conjunto de teste |
| **Accuracy** | Proporção de previsões corretas |
| **Balanced Accuracy** | Accuracy média por classe (robusta a desequilíbrios) |
| **Macro F1-Score** | Média não ponderada do F1 por classe (métrica principal) |
| **Weighted F1-Score** | Média ponderada do F1 por classe |
| **Macro Precision** | Média não ponderada da Precision por classe |
| **Macro Recall** | Média não ponderada do Recall por classe |

Os notebooks geram ainda automaticamente: curvas de treino/validação, matriz de confusão, F1 por classe e gráfico comparativo entre os dois modelos.

---

## Como Executar

### Pré-requisitos

```bash
pip install numpy pandas scikit-learn matplotlib torch
```

### Ordem de execução

```
1_Treino_Numpy.ipynb   →   gera modelo_numpy_artefactos.pkl
2_Treino_PyTorch.ipynb →   gera modelo_pytorch_gru.pth + pytorch_vocab.pkl
3_Submissao_Final.ipynb →  gera resultados.csv
```

> **Nota:** O notebook 2 e o notebook 3 dependem de `modelo_numpy_artefactos.pkl` para carregar o `LabelEncoder` oficial. O notebook 1 deve sempre ser corrido primeiro.

---

## Como Correr a Avaliação (Docente)

1. Abrir o ficheiro **`3_Submissao_Final.ipynb`**.
2. *(Opcional)* Alterar a variável `PATH_DATASET_TESTE` na primeira célula para apontar para o CSV de validação correto. O formato esperado é `ID;Text` (separador `;`).
3. Clicar em **"Run All"**.
4. O resultado será gerado no ficheiro **`resultados.csv`** com colunas:

| Coluna | Descrição |
|--------|-----------|
| `ID` | Identificador do exemplo |
| `Text` | Texto original |
| `Predict_Numpy_DNN` | Previsão do modelo NumPy |
| `Predict_PyTorch_GRU` | Previsão do modelo PyTorch GRU |



