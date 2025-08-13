# Análise Multimodal de Mídia com IA

Este repositório contém dois projetos de Inteligência Artificial para análise de conteúdo multimídia:

1.  **Analisador de Áudio**: Um script que utiliza o modelo **Whisper da OpenAI** para transcrever áudio para texto e, em seguida, aplica técnicas de Processamento de Linguagem Natural (PLN) para analisar a frequência das palavras.
2.  **Classificador de Vídeo**: Um pipeline completo de Visão Computacional que extrai frames de um vídeo, os processa, e treina uma **Rede Neural Convolucional (CNN)** com TensorFlow/Keras para classificar objetos contidos nos frames (livro, faca, celular).

## 🎬 Vídeo Original

Este é o vídeo utilizado como fonte de dados para o projeto de Visão Computacional.

[![Assista ao Vídeo]](https://youtube.com/shorts/at-nwAkwYv0)

## 🚀 Funcionalidades Principais

### Análise de Áudio
-   Transcrição de áudio em português para texto.
-   Limpeza e pré-processamento de texto (tokenização, remoção de stopwords, lematização).
-   Contagem e visualização das palavras mais frequentes em um gráfico de barras.

### Análise de Vídeo
-   Separação automática de áudio e vídeo de um arquivo `.mp4`.
-   Extração de frames de vídeo em intervalos regulares.
-   Armazenamento de frames como imagens (`.jpg`) e em formato FITS (`.fits`), com metadados.
-   Gestão de metadados dos frames em um banco de dados **SQLite**.
-   Pré-processamento de imagem, incluindo equalização de histograma e segmentação de cores com K-Means.
-   Técnicas de balanceamento de dados (`RandomUnderSampler`).
-   *Data Augmentation* para enriquecer o dataset de treino (rotações, zoom, brilho, etc.).
-   Treinamento e avaliação de uma CNN para classificação de imagens multiclasse.
-   Visualização detalhada dos resultados do modelo:
    -   Métricas de acurácia e perda.
    -   Relatório de classificação (precisão, recall, F1-score).
    -   Matriz de Confusão.
    -   Curva ROC.
    -   Visualização dos filtros aprendidos pela CNN.

## 📁 Estrutura do Projeto

```
.
├── Analisando_áudio.ipynb      # Notebook para transcrição e análise de áudio.
├── Analisando_Vídeo.ipynb      # Notebook para extração de frames e treinamento do modelo de visão.
├── requeriments.txt            # Lista de todas as dependências Python.
├── Video visao computacional.mp4  # (Opcional) Vídeo de entrada para o projeto de visão.
│
├── audio_formatado.mp3         # (Gerado) Áudio extraído pelo script de vídeo.
├── video_formatado.mp4         # (Gerado) Vídeo sem áudio.
├── base_frames.db              # (Gerado) Banco de dados SQLite com informações dos frames.
├── frames_video/               # (Gerado) Pasta com os frames extraídos em formato JPG.
├── frames_video_fits/          # (Gerado) Pasta com os frames extraídos em formato FITS.
└── *.keras                     # (Gerado) Modelos treinados salvos pelo Keras.
```

---

## 🔧 Pré-requisitos

Antes de começar, garanta que você tenha os seguintes softwares instalados:
-   [Python 3.10+](https://www.python.org/downloads/)
-   [Pip](https://pip.pypa.io/en/stable/installation/) (geralmente vem com o Python)
-   [Git](https://git-scm.com/downloads/)

## ⚙️ Passo a Passo da Instalação

Siga estes passos para configurar seu ambiente de desenvolvimento.

### 1. Clonar o Repositório

Abra seu terminal ou prompt de comando e clone este repositório:
```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

### 2. Criar e Ativar um Ambiente Virtual

É altamente recomendável usar um ambiente virtual para isolar as dependências do projeto e evitar conflitos.

```bash
# Criar o ambiente virtual (pode usar 'venv' ou o nome que preferir)
python -m venv venv

# Ativar o ambiente virtual
# No Windows:
venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate
```
Você saberá que o ambiente está ativo quando vir `(venv)` no início do seu prompt de comando.

### 3. Instalar as Dependências

Com o ambiente virtual ativado, instale todas as bibliotecas necessárias usando o arquivo `requeriments.txt`.

```bash
pip install -r requeriments.txt
```
Este comando instalará todas as bibliotecas, incluindo `tensorflow`, `pandas`, `openai-whisper`, `nltk`, `moviepy`, entre outras.

### 4. Baixar Pacotes NLTK

O projeto de análise de áudio utiliza a biblioteca NLTK. Após a instalação, é necessário baixar alguns pacotes de dados específicos para o processamento de texto em português.

O notebook `Analisando_áudio.ipynb` já contém o código para fazer isso. Ao executar a primeira célula de código, ele tentará baixar os seguintes pacotes:
-   `punkt` (para tokenização)
-   `stopwords` (para lista de palavras de parada)
-   `wordnet` (para lematização)

Caso encontre algum problema, você pode executar o download manualmente em um script Python:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## ▶️ Como Executar os Modelos

Os dois projetos estão em notebooks Jupyter (`.ipynb`). Você pode abri-los usando o Jupyter Lab, Jupyter Notebook ou uma IDE como o VS Code com a extensão Jupyter.

### Parte 1: Análise e Classificação de Vídeo (`Analisando_Vídeo.ipynb`)

Este é o primeiro notebook a ser executado, pois ele **gera o arquivo de áudio** necessário para a segunda parte.

#### **Passo 1: Preparar o Vídeo de Entrada**
-   O notebook está configurado para baixar um vídeo de exemplo do Google Drive usando o comando `gdown`.
-   Alternativamente, você pode colocar seu próprio arquivo de vídeo no formato `.mp4` na pasta raiz do projeto. Se usar seu próprio vídeo, certifique-se de que o nome do arquivo seja **`Video visao computacional.mp4`** ou altere o nome no código.

#### **Passo 2: Executar as Células do Notebook**
Abra o notebook `Analisando_Vídeo.ipynb` e execute as células em ordem.

1.  **Imports e Download**: A primeira célula instala `moviepy` e baixa o vídeo de exemplo.
2.  **Separando áudio e vídeo**: Esta célula é crucial. Ela irá criar dois novos arquivos na pasta raiz:
    -   `audio_formatado.mp3` (será usado no outro notebook)
    -   `video_formatado.mp4` (vídeo sem áudio, para processamento)
3.  **Criando base de dados e Extraindo frames**: Esta seção criará as pastas `frames_video/`, `frames_video_fits/` e o arquivo `base_frames.db`. Este processo pode levar alguns minutos, dependendo da duração e resolução do vídeo.
4.  **Analisando imagem**: Demonstra como carregar uma imagem do banco de dados e aplicar a equalização de histograma.
5.  **Rotulando os dados**: O código rotula os frames no banco de dados com base em intervalos pré-definidos (`book`, `knife`, `cell phone`).
6.  **Preparando dados para o modelo**: Esta seção carrega os dados para um DataFrame Pandas, aplica filtros e realiza o balanceamento e o aumento de dados (*Data Augmentation*).
7.  **Treinando o modelo**: A célula `model.fit` iniciará o treinamento da Rede Neural.
    -   ⚠️ **Atenção**: O treinamento de uma CNN é um processo computacionalmente intensivo. Recomenda-se uma máquina com uma **GPU dedicada** (NVIDIA com CUDA) para acelerar o processo. Sem uma GPU, o treinamento pode levar várias horas.
    -   Durante o treinamento, modelos de checkpoint (`.keras`) serão salvos no diretório.
8.  **Analisando o resultado**: As células finais irão carregar o melhor modelo salvo e gerar gráficos e métricas de avaliação, como a matriz de confusão e a curva ROC.

### Parte 2: Análise de Áudio (`Analisando_áudio.ipynb`)

Este notebook deve ser executado após a **Parte 1**, pois depende do arquivo de áudio gerado.

#### **Passo 1: Verificar o Arquivo de Áudio**
-   Certifique-se de que o arquivo `audio_formatado.mp3` existe na pasta raiz do projeto. Ele deve ter sido criado ao executar a célula "Separando áudio e vídeo" do notebook `Analisando_Vídeo.ipynb`.

#### **Passo 2: Executar as Células do Notebook**
Abra o notebook `Analisando_áudio.ipynb` e execute as células em ordem.

1.  **Imports**: Carrega todas as bibliotecas e baixa os pacotes NLTK, se necessário.
2.  **Carregando modelo**: Carrega o modelo "base" do Whisper. O primeiro download pode demorar um pouco.
3.  **Transcrevendo áudio**: Lê o arquivo `audio_formatado.mp3` e exibe o texto transcrito.
4.  **Tratando o texto**: Realiza o pré-processamento do texto (tokenização e lematização).
5.  **Contagem de Palavras**: Conta a frequência de cada palavra e exibe as 10 mais comuns.
6.  **Plot principais palavras**: Gera um gráfico de barras visualizando as palavras mais frequentes.

## 📊 Entendendo os Resultados

Após a execução completa de ambos os notebooks, você terá os seguintes artefatos:

-   **Texto Transcrito**: A saída da transcrição do áudio no notebook de áudio.
-   **Gráfico de Frequência**: Uma visualização das palavras mais comuns no áudio.
-   **Banco de Dados de Frames (`base_frames.db`)**: Um arquivo SQLite contendo metadados e os dados brutos de cada frame extraído.
-   **Modelo Treinado (`.keras`)**: O arquivo contendo a arquitetura e os pesos da CNN treinada, pronto para ser usado para novas predições.
-   **Métricas de Avaliação**: Gráficos e relatórios detalhados no notebook de vídeo que mostram o quão bem o modelo de classificação de imagens performou.
