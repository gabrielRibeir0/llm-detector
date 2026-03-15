# UC Aprendizagem Profunda - Trabalho Prático Módulo 1
## Grupo [Inserir Número do Grupo]

### Elementos do Grupo
- Nome 1 (AXXXXX)
- Nome 2 (AXXXXX)
- Nome 3 (AXXXXX)

### Organização do Repositório
O repositório está organizado da seguinte forma:
- `data/`: Contém os datasets (treino e teste cego).
- `src/`: Contém o framework desenvolvido inteiramente em NumPy para a Tarefa 2 (Redes Neuronais sem bibliotecas externas).
- `1_Treino_Numpy.ipynb`: Notebook com o treino do modelo de Deep Neural Network customizado (Numpy). Usa TF-IDF para features.
- `2_Treino_PyTorch.ipynb`: Notebook com o treino do modelo RNN Bidirecional (GRU) utilizando PyTorch.
- `3_Submissao_Final.ipynb`: **Notebook de avaliação.** Carrega o dataset de teste, inicializa as arquiteturas e os pesos guardados (em `.pkl` e `.pth`) e gera um CSV com as previsões.

### Como correr a avaliação (Docente)
1. Abrir o ficheiro `3_Submissao_Final.ipynb`.
2. Opcional: Alterar a variável `PATH_DATASET_TESTE` para apontar para o CSV de validação correto (formatado com colunas `ID;Text`).
3. Clicar em "Run All".
4. O resultado será gerado no ficheiro `resultados_grupoX.csv`.