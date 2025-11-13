# Deep Learning para Previsão de Biomassa (Kaggle CSIRO)

## Objetivo do Projeto

Este projeto implementa uma solução de ponta a ponta para a competição do Kaggle **"CSIRO - Image2Biomass Prediction"**. O objetivo é desenvolver um modelo de Deep Learning (Visão Computacional) capaz de prever cinco componentes de biomassa (ex: `Dry_Green_g`, `Dry_Clover_g`) analisando imagens aéreas de pastagens.

Este repositório documenta a evolução de um modelo de *baseline* (base) para uma arquitetura SOTA (Estado-da-Arte), detalhando o pipeline de dados e as estratégias de treino.

* **Competição:** [CSIRO - Image2Biomass Prediction](https://www.kaggle.com/competitions/csiro-biomass)

## Stack de Tecnologias

* **Linguagem:** Python
* **Bibliotecas de Dados:** Pandas, NumPy, Scikit-learn
* **Frameworks de Deep Learning:** TensorFlow/Keras e PyTorch/Transformers (Hugging Face)
* **Outras:** Kaggle API, Streamlit (para a app de protótipo)

## Pipeline de Dados (O Desafio do Formato)

O primeiro desafio técnico foi o pré-processamento dos dados. O `train.csv` fornecido estava em formato "longo", com 5 linhas para cada imagem (uma por alvo). Para o treino, era necessário um formato "largo" (uma linha por imagem, com 5 colunas de alvos).

**Ação (ETL):**
1.  **Carregar:** Ler o `train.csv` (1785 linhas).
2.  **Pivotar:** Usar `pandas.pivot_table` para transformar os dados.
    * `index='image_path'`
    * `columns='target_name'`
    * `values='target'`
3.  **Juntar:** Combinar os alvos "pivotados" com os metadados originais (como `State`, `Species`).
4.  **Salvar:** O resultado é um `train_processed_WIDE.csv` limpo (357 linhas), que se torna a nossa "fonte da verdade" para o treino.



## Metodologia de Modelagem e Evolução

Com um dataset de treino muito pequeno (357 imagens), o risco de **overfitting** (sobreajuste) era o inimigo principal. A estratégia evoluiu em duas fases:

### Fase 1: Baseline com EfficientNetB0 (Score: 0.40)

O primeiro modelo foi construído em **TensorFlow/Keras** para estabelecer uma baseline robusta.

* **Modelo:** `EfficientNetB0` (pré-treinado em ImageNet).
* **Técnica:** Transfer Learning (Aprendizagem por Transferência)
* **Estratégia de Treino:**
    1.  **Treino da "Cabeça" (Head):** O `base_model` foi "congelado" (`trainable=False`) e apenas uma nova "cabeça" de regressão foi treinada por 60 épocas.
    2.  **Callbacks:** `ModelCheckpoint` foi usado para salvar apenas o melhor modelo, e `EarlyStopping` para prevenir treino desnecessário.
* **Resultado:** Esta abordagem foi bem-sucedida e alcançou um score de **0.40 $R^2$** no leaderboard público.

### Fase 2: Modelo Avançado com DINOv2 (Score: >0.50)

Para bater a baseline de 0.40, o projeto foi migrado para uma arquitetura SOTA: **DINOv2 (Vision Transformer)**, um modelo que aprendeu representações visuais robustas através de auto-supervisão.

* **Framework:** `PyTorch` & `Transformers (Hugging Face)`.
* **Modelo:** `metaresearch/dinov2` (PyTorch/base).
* **Técnica de Combate ao Overfitting:**
    1.  **Data Augmentation:** Implementei `torchvision.transforms` (como `RandomHorizontalFlip`, `RandomRotation`, `RandomVerticalFlip`) para "mexer" nas 303 imagens de treino, criando dados novos em cada época.
    2.  **Fine-Tuning em Duas Fases:**
        * **Fase A (Treino da Cabeça):** Treino de uma "cabeça" de regressão em PyTorch com o `dino_base` congelado (`requires_grad=False`).
        * **Fase B (Fine-Tuning):** O `dino_base` foi "descongelado" e treinado com uma taxa de aprendizagem (learning rate) muito baixa (`1e-6`), permitindo ao modelo adaptar-se ao nosso problema específico sem "esquecer" o seu conhecimento prévio.

<img width="1223" height="649" alt="previsor" src="https://github.com/user-attachments/assets/4e2d97a9-e750-4dd4-a275-3fab653e4f6b" />


## Resultados

* **Modelo Baseline (EfficientNetB0):** 0.40 $R^2$
* **Modelo Avançado (DINOv2):** `[Insira o seu novo score aqui, ex: 0.52]`

O DINOv2 provou ser um backbone superior, e a combinação de Data Augmentation e Fine-Tuning foi crucial para gerir o dataset pequeno e melhorar o score da baseline.

## Como Executar

Este projeto existe em dois locais:

### 1. A Aplicação de Portfólio (Streamlit)

Uma app interativa que demonstra o conceito de *multi-input* (combinando imagens com metadados).

```bash
# 1. Clone o repositório (se estiver no GitHub)
git clone [SEU_LINK_GITHUB]
cd [NOME_DA_PASTA]

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute a app
streamlit run app.py
```
