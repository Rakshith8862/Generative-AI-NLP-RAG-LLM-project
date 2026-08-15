# Generative-AI-NLP-RAG-LLM-project
End-to-end Generative AI and NLP project covering sentiment analysis with RNN/LSTM, Seq2Seq translation, Transformers, sentence embeddings, Llama, PDF-based RAG, Gemini, Tavily, AI fact checking, and travel planning.

# 🤖 Generative AI & NLP — End-to-End AI Project

A comprehensive hands-on **Generative AI, Natural Language Processing, Deep Learning, LLM, RAG, and AI Agent project** implemented using Python, TensorFlow, PyTorch, Hugging Face, LangChain, Llama.cpp, Google Gemini, Tavily, and Sentence Transformers.

This repository contains multiple interconnected experiments and applications developed as part of an end-to-end exploration of modern AI techniques.

The project starts with traditional NLP and deep-learning-based sentiment analysis and gradually progresses toward:

- Text preprocessing
- Binary sentiment classification
- RNN
- LSTM
- Hyperparameter search
- Model checkpointing
- Model persistence
- Seq2Seq Encoder-Decoder
- English → Hindi translation
- Transformer architecture
- Sentence embeddings
- Local Llama models
- PDF document question answering
- Retrieval-Augmented Generation (RAG)
- Google Gemini API
- Tavily web search
- AI-powered fact checking
- AI travel planning assistant

---

# 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Project Objectives](#-project-objectives)
- [What This Project Contains](#-what-this-project-contains)
- [Project Architecture](#-project-architecture)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Environment Requirements](#-environment-requirements)
- [Installation](#-installation)
- [Important API Key Security](#-important-api-key-security)
- [Module 1 — Amazon Fine Food Review Sentiment Analysis](#module-1--amazon-fine-food-review-sentiment-analysis)
- [1. Dataset Download](#1-dataset-download)
- [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
- [3. Missing Value Analysis](#3-missing-value-analysis)
- [4. Review Rating Distribution](#4-review-rating-distribution)
- [5. Text Analysis](#5-text-analysis)
- [6. Binary Sentiment Classification](#6-binary-sentiment-classification)
- [7. Class Imbalance](#7-class-imbalance)
- [8. Data Balancing](#8-data-balancing)
- [9. Text Cleaning](#9-text-cleaning)
- [10. Tokenization](#10-tokenization)
- [11. Padding](#11-padding)
- [12. Train-Test Split](#12-train-test-split)
- [Module 2 — RNN Sentiment Classifier](#module-2--rnn-sentiment-classifier)
- [RNN Architecture](#rnn-architecture)
- [RNN Training](#rnn-training)
- [RNN Result](#rnn-result)
- [Module 3 — LSTM Sentiment Classifier](#module-3--lstm-sentiment-classifier)
- [LSTM Architecture](#lstm-architecture)
- [LSTM Training](#lstm-training)
- [LSTM Result](#lstm-result)
- [Module 4 — Hyperparameter Search](#module-4--hyperparameter-search)
- [Search Space](#search-space)
- [Random Search](#random-search)
- [Best Model](#best-model)
- [Module 5 — Final Sentiment Model](#module-5--final-sentiment-model)
- [Model Checkpoint](#model-checkpoint)
- [Saving the Tokenizer](#saving-the-tokenizer)
- [Loading the Model](#loading-the-model)
- [Testing Sentiment Prediction](#testing-sentiment-prediction)
- [Module 6 — English to Hindi Seq2Seq Translation](#module-6--english-to-hindi-seq2seq-translation)
- [Translation Dataset](#translation-dataset)
- [Vocabulary Creation](#vocabulary-creation)
- [Sequence Preparation](#sequence-preparation)
- [Encoder](#encoder)
- [Decoder](#decoder)
- [Seq2Seq Model](#seq2seq-model)
- [Training](#training)
- [Translation Function](#translation-function)
- [Translation Results](#translation-results)
- [Module 7 — Tiny Transformer](#module-7--tiny-transformer)
- [Module 8 — Sentence Transformers](#module-8--sentence-transformers)
- [Module 9 — Local Llama + PDF RAG](#module-9--local-llama--pdf-rag)
- [Installing Llama.cpp](#installing-llamacpp)
- [Downloading the Llama Model](#downloading-the-llama-model)
- [Uploading a PDF](#uploading-a-pdf)
- [Loading the PDF](#loading-the-pdf)
- [Creating Embeddings](#creating-embeddings)
- [Vector Store](#vector-store)
- [RAG Pipeline](#rag-pipeline)
- [Asking Questions](#asking-questions)
- [Module 10 — Google Gemini API](#module-10--google-gemini-api)
- [Gemini Setup](#gemini-setup)
- [Gemini Example](#gemini-example)
- [Module 11 — Tavily Web Search](#module-11--tavily-web-search)
- [Module 12 — AI Fact Checker](#module-12--ai-fact-checker)
- [Fact Checker Workflow](#fact-checker-workflow)
- [Fact Checker Output](#fact-checker-output)
- [Module 13 — Travel Planning Assistant](#module-13--travel-planning-assistant)
- [Travel Assistant Workflow](#travel-assistant-workflow)
- [Travel Research](#travel-research)
- [Travel Plan Generation](#travel-plan-generation)
- [Travel Alerts](#travel-alerts)
- [Saving the Travel Plan](#saving-the-travel-plan)
- [How to Run the Complete Project](#-how-to-run-the-complete-project)
- [Running in Google Colab](#-running-in-google-colab)
- [Running Locally](#-running-locally)
- [Expected Outputs](#-expected-outputs)
- [Important Implementation Notes](#-important-implementation-notes)
- [Known Issues and Limitations](#-known-issues-and-limitations)
- [Recommended Improvements](#-recommended-improvements)
- [Learning Outcomes](#-learning-outcomes)
- [Future Improvements](#-future-improvements)
- [Conclusion](#-conclusion)
- [Author](#-author)

---

# 📖 Project Overview

This project is a practical exploration of **Generative AI and Natural Language Processing**.

Rather than implementing only one model, the notebook progresses through several levels of AI development.

The overall progression is:

```text
Raw Text
   ↓
Text Cleaning
   ↓
Tokenization
   ↓
Sequence Modeling
   ↓
RNN
   ↓
LSTM
   ↓
Hyperparameter Search
   ↓
Model Saving / Loading
   ↓
Seq2Seq
   ↓
Transformer
   ↓
Sentence Embeddings
   ↓
Local LLM
   ↓
PDF RAG
   ↓
Gemini API
   ↓
Web Search
   ↓
Fact Checking
   ↓
Travel Planning Agent
```

The project therefore demonstrates both **classical NLP/deep learning workflows** and more modern **Generative AI / LLM workflows**.

---

# 🎯 Project Objectives

The main objectives of the project are:

1. Understand NLP data preprocessing.
2. Analyze Amazon customer reviews.
3. Convert star ratings into binary sentiment.
4. Handle class imbalance.
5. Clean natural-language text.
6. Convert text into numerical sequences.
7. Build an RNN sentiment classifier.
8. Build an LSTM sentiment classifier.
9. Compare RNN and LSTM behavior.
10. Perform random hyperparameter search.
11. Select a high-performing configuration.
12. Save and reload a trained deep-learning model.
13. Build an English-to-Hindi Seq2Seq model.
14. Understand Encoder-Decoder architecture.
15. Experiment with a small Transformer.
16. Generate sentence embeddings.
17. Run a local Llama model.
18. Build a PDF-based RAG pipeline.
19. Use vector embeddings for document retrieval.
20. Use Google Gemini for generation.
21. Use Tavily for web search.
22. Build an AI fact-checking workflow.
23. Build an AI travel-planning assistant.

---

# 🧩 What This Project Contains

The notebook contains the following major modules:

| Module | Topic |
|---|---|
| 1 | Amazon Fine Food Review Analysis |
| 2 | RNN Sentiment Classification |
| 3 | LSTM Sentiment Classification |
| 4 | Hyperparameter Search |
| 5 | Final Sentiment Model |
| 6 | English → Hindi Seq2Seq |
| 7 | Tiny Transformer |
| 8 | Sentence Transformers |
| 9 | Local Llama + PDF RAG |
| 10 | Google Gemini API |
| 11 | Tavily Web Search |
| 12 | AI Fact Checker |
| 13 | Travel Planning Assistant |

---

# 🏗️ Project Architecture

The complete project can be visualized as:

```text
                    GENERATIVE AI / NLP PROJECT
                              |
          +-------------------+-------------------+
          |                   |                   |
          ↓                   ↓                   ↓
     NLP / DL              LLM / RAG          AI Agents
          |                   |                   |
          ↓                   ↓                   ↓
 Amazon Reviews          Local Llama           Fact Checker
          |                   |                   |
          ↓                   ↓                   ↓
 RNN / LSTM              PDF Loader          Travel Buddy
          |                   |
          ↓                   ↓
 Hyperparameter          Embeddings
 Search                     |
          |                   ↓
          ↓               Vector Store
 Sentiment Model             |
                              ↓
                            RAG
                              |
                              ↓
                         Llama LLM
```

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Data Processing

- NumPy
- Pandas

## Visualization

- Matplotlib

## Machine Learning

- Scikit-learn

## Deep Learning

- TensorFlow
- Keras
- PyTorch

## NLP

- Keras Tokenizer
- Sequence Padding
- RNN
- LSTM
- Seq2Seq
- Encoder-Decoder
- Transformer
- Sentence Transformers

## Generative AI

- Google Gemini
- Llama 3.2 3B Instruct
- Llama.cpp

## Retrieval-Augmented Generation

- LangChain
- PDF Loader
- Sentence Transformers
- DocArray
- Vector Store

## Web Search

- Tavily

## Dataset

- Kaggle Amazon Fine Food Reviews

## Environment

- Google Colab
- Jupyter Notebook
- Python 3

---

# 📁 Project Structure

A recommended GitHub repository structure is:

```text
GenAI-NLP-Project/
│
├── Gen_ai_ff_final.ipynb
│
├── README.md
│
├── requirements.txt
│
├── models/
│   ├── best_amazon_sentiment.h5
│   └── best_amazon_sentiment.pkl
│
├── data/
│   └── README.md
│
├── documents/
│   └── README.md
│
└── .gitignore
```

The current notebook itself contains the complete workflow.

---

# 💻 Environment Requirements

Recommended:

```text
Python 3.10+
```

For the original notebook, **Google Colab is recommended**, particularly for:

- TensorFlow training
- GPU acceleration
- Llama.cpp
- Downloading Hugging Face models
- Running the PDF RAG example

A GPU is helpful but not mandatory for every part of the project.

---

# 📦 Installation

Install the major Python dependencies:

```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install tensorflow
pip install torch
pip install kagglehub
pip install sentence-transformers
pip install transformers
pip install huggingface-hub
pip install langchain
pip install langchain-community
pip install pypdf
pip install docarray
pip install llama-cpp-python
pip install google-generativeai
pip install tavily-python
```

The notebook also installs several packages directly inside cells.

For the local Llama/RAG section, the notebook uses:

```bash
pip install langchain langchain-community pypdf docarray sentence-transformers huggingface_hub llama-cpp-python
```

It also installs build dependencies and clones/builds Llama.cpp in the original Colab workflow.

---

# 🔐 Important API Key Security

## ⚠️ READ THIS BEFORE UPLOADING THE NOTEBOOK TO GITHUB

The original notebook contains API credentials directly inside Python code.

For example, the notebook configures:

```python
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

and:

```python
TavilyClient(api_key="YOUR_TAVILY_API_KEY")
```

Do **not** publish real API keys in a public GitHub repository.

### If you already uploaded the notebook containing real keys:

1. Immediately revoke the exposed Gemini key.
2. Revoke the exposed Tavily key.
3. Generate new keys.
4. Remove the old keys from the notebook.
5. Remove them from Git history if necessary.
6. Use environment variables or a `.env` file.

---

# 🔑 Recommended API Key Setup

Create a `.env` file:

```text
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Install:

```bash
pip install python-dotenv
```

Then:

```python
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
```

For Google Gemini:

```python
import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)
```

For Tavily:

```python
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
```

Add this to `.gitignore`:

```text
.env
```

---

# 🧪 MODULE 1 — Amazon Fine Food Review Sentiment Analysis

The first major section uses the **Amazon Fine Food Reviews dataset**.

The notebook downloads the dataset through:

```python
path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
```

and loads:

```python
Reviews.csv
```

The original dataset contains:

```text
568,454 reviews
10 columns
```

The columns are:

```text
Id
ProductId
UserId
ProfileName
HelpfulnessNumerator
HelpfulnessDenominator
Score
Time
Summary
Text
```

The notebook confirms:

```text
Total Reviews : 568454
```

---

# 1. Dataset Download

The notebook uses KaggleHub:

```python
import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download(
    "snap/amazon-fine-food-reviews"
)

csv_file = os.path.join(
    path,
    "Reviews.csv"
)

df = pd.read_csv(csv_file)
```

This means the dataset does not need to be manually downloaded if KaggleHub is correctly configured.

---

# 2. Exploratory Data Analysis

The notebook examines:

```python
df.head(2)
```

and:

```python
df.shape
df.columns.to_list()
```

The first records contain information such as:

```text
Product ID
User ID
Profile Name
Helpfulness
Score
Time
Summary
Text
```

The project focuses mainly on:

```text
Text
Score
```

for sentiment analysis.

---

# 3. Missing Value Analysis

The notebook runs:

```python
df.info()
df.isnull().sum()
```

The missing values reported are:

| Column | Missing |
|---|---:|
| Id | 0 |
| ProductId | 0 |
| UserId | 0 |
| ProfileName | 26 |
| HelpfulnessNumerator | 0 |
| HelpfulnessDenominator | 0 |
| Score | 0 |
| Time | 0 |
| Summary | 27 |
| Text | 0 |

Because the project only uses `Text` and `Score`, it removes rows missing those fields:

```python
df_clean = df[['Text', 'Score']].dropna()
```

All 568,454 reviews remain after this cleaning step.

---

# 4. Review Rating Distribution

The dataset contains five ratings:

```text
1 star
2 stars
3 stars
4 stars
5 stars
```

The distribution recorded in the notebook is:

| Rating | Reviews | Percentage |
|---:|---:|---:|
| 1 | 52,268 | 9.19% |
| 2 | 29,769 | 5.24% |
| 3 | 42,640 | 7.50% |
| 4 | 80,655 | 14.19% |
| 5 | 363,122 | 63.88% |

The dataset is therefore strongly dominated by **5-star reviews**.

---

# 5. Text Analysis

The project creates two additional text features:

```python
df_clean['text_length'] = df_clean["Text"].str.len()

df_clean['word_count'] = (
    df_clean['Text']
    .str.split()
    .str.len()
)
```

These features are used for exploratory analysis.

The notebook visualizes:

1. Rating distribution
2. Text-length distribution
3. Word-count distribution

The project uses Matplotlib for visualization.

---

# 6. Binary Sentiment Classification

The original rating variable contains five classes.

The project converts it into a binary sentiment problem.

Three-star reviews are removed:

```python
df_binary = df_clean[
    df_clean['Score'] != 3
].copy()
```

The sentiment label is then created:

```python
df_binary['sentiment'] = (
    df_binary['Score'] >= 4
).astype(int)
```

The mapping is:

```text
1–2 stars → Negative → 0

4–5 stars → Positive → 1
```

Therefore:

```text
0 = Negative
1 = Positive
```

---

# 7. Class Imbalance

After removing 3-star reviews:

```text
Negative reviews = 82,037
Positive reviews = 443,777
```

Total binary reviews:

```text
525,814
```

Approximate distribution:

```text
Negative = 15.60%
Positive = 84.40%
```

This is a highly imbalanced dataset.

The notebook identifies this using:

```python
if positive_pct > 70:
    print("Dataset is imbalanced!")
```

---

# 8. Data Balancing

To handle the imbalance, the project performs **random undersampling of the majority class**.

Negative reviews are selected:

```python
negative_reviews = df_binary[
    df_binary['sentiment'] == 0
]
```

Positive reviews are selected:

```python
positive_reviews = df_binary[
    df_binary['sentiment'] == 1
]
```

The minority class contains:

```text
82,037
```

positive reviews are therefore downsampled to the same size:

```python
positive_downsampled = resample(
    positive_reviews,
    replace=False,
    n_samples=n_minority,
    random_state=21
)
```

The final balanced dataset contains approximately:

```text
Negative = 82,037
Positive = 82,037
Total    = 164,074
```

The dataset is shuffled:

```python
df_balanced = df_balanced.sample(
    frac=1,
    random_state=21
).reset_index(drop=True)
```

---

# 9. Text Cleaning

The project defines:

```python
def clean_text(text):
    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    text = ' '.join(text.split())

    return text
```

The preprocessing performs:

1. Lowercasing
2. Removal of non-alphabetic characters
3. Removal of numbers and special characters
4. Whitespace normalization

Example:

```text
Original:
I LOVE this product!!! 100%

Processed:
i love this product
```

The cleaned text is stored in:

```python
df_balanced['clean_text']
```

---

# 10. Tokenization

The project uses the Keras Tokenizer.

Configuration:

```python
MAX_FEATURES = 10000
MAX_LEN = 100
```

Tokenizer:

```python
tokenizer = Tokenizer(
    num_words=MAX_FEATURES,
    oov_token='<OOV>'
)
```

It is fitted using:

```python
tokenizer.fit_on_texts(X)
```

Then text is converted into integer sequences:

```python
X_sequences = tokenizer.texts_to_sequences(X)
```

---

# 11. Padding

Since neural networks require consistent input lengths, the sequences are padded:

```python
X_padded = pad_sequences(
    X_sequences,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)
```

The configuration is:

```text
Maximum vocabulary = 10,000 words
Maximum sequence length = 100 words
```

Long reviews are truncated.

Short reviews are padded.

---

# 12. Train-Test Split

The dataset is split using:

```python
train_test_split(
    X_padded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The split is:

```text
80% Training
20% Testing
```

For the balanced dataset of 164,074 samples, this corresponds approximately to:

```text
Training = 131,259
Testing  = 32,815
```

The use of:

```python
stratify=y
```

maintains the class distribution in the training and test sets.

---

# 🤖 MODULE 2 — RNN Sentiment Classifier

The first deep-learning model is a **Simple Recurrent Neural Network**.

---

# RNN Architecture

The architecture is:

```text
Input Text
    ↓
Tokenizer
    ↓
Embedding
    ↓
SimpleRNN
    ↓
Dropout
    ↓
Dense
    ↓
Sigmoid
    ↓
Positive / Negative
```

The actual model is:

```python
Sequential([
    Embedding(
        input_dim=MAX_FEATURES,
        output_dim=128,
        input_length=MAX_LEN
    ),

    SimpleRNN(
        units=64,
        return_sequences=False
    ),

    Dropout(0.5),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])
```

The model uses:

```text
Embedding dimension = 128
RNN units = 64
Dropout = 0.5
Dense units = 32
Output = 1 sigmoid neuron
```

---

# RNN Training

The model is compiled with:

```python
optimizer='adam'
loss='binary_crossentropy'
metrics=['accuracy']
```

Training configuration:

```text
Batch size = 128
Epochs = 5
Validation split = 20%
```

---

# RNN Result

Recorded training result:

```text
Final training accuracy ≈ 67.33%
```

The validation accuracy fluctuated during training.

The recorded validation accuracy by epoch was approximately:

```text
Epoch 1 → 55.89%
Epoch 2 → 55.72%
Epoch 3 → 63.23%
Epoch 4 → 58.25%
Epoch 5 → 58.64%
```

The RNN therefore performed considerably worse than the later LSTM model in this experiment.

---

# 🤖 MODULE 3 — LSTM Sentiment Classifier

The second deep-learning model is an **LSTM**.

LSTM is designed to handle sequential dependencies more effectively than a basic RNN.

---

# LSTM Architecture

The architecture is:

```text
Input
 ↓
Embedding
 ↓
LSTM
 ↓
Dropout
 ↓
Dense
 ↓
Sigmoid
 ↓
Sentiment
```

The model is:

```python
Sequential([
    Embedding(
        input_dim=MAX_FEATURES,
        output_dim=128,
        input_length=MAX_LEN
    ),

    LSTM(
        units=64,
        return_sequences=False
    ),

    Dropout(0.5),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])
```

---

# LSTM Training

Configuration:

```text
Optimizer = Adam
Loss = Binary Crossentropy
Batch size = 128
Epochs = 5
Validation split = 20%
```

---

# LSTM Result

The recorded validation accuracy was:

```text
Epoch 1 → 55.61%
Epoch 2 → 82.95%
Epoch 3 → 89.28%
Epoch 4 → 90.40%
Epoch 5 → 91.40%
```

The final recorded training accuracy was approximately:

```text
93.22%
```

The LSTM substantially outperformed the basic RNN in this experiment.

---

# 🔬 MODULE 4 — Hyperparameter Search

After experimenting with RNN and LSTM, the project performs a random hyperparameter search.

The search explores:

- Model type
- Embedding dimension
- Number of recurrent units
- Dropout
- Learning rate

---

# Search Space

The parameter grid is:

```python
param_grid = {
    'model_type': ['rnn', 'lstm'],
    'embedding_dim': [64, 128],
    'units': [32, 64, 128],
    'dropout_rate': [0.30, 0.50],
    'learning_rate': [0.001, 0.01]
}
```

The theoretical number of combinations is:

```text
2 × 2 × 3 × 2 × 2 = 48
```

The notebook does not evaluate all 48 combinations.

Instead, it performs:

```text
8 random trials
```

---

# 🎲 Random Search

The search function randomly selects parameters:

```python
def random_search(n_trails=8):
```

Each trial trains the selected model for:

```text
10 epochs
```

using:

```text
Batch size = 128
Validation split = 20%
```

The best validation accuracy from each trial is stored.

---

# 🏆 Random Search Results

The recorded results were:

| Trial | Model | Validation Accuracy |
|---:|---|---:|
| 1 | RNN | 79.02% |
| 2 | LSTM | **91.81%** |
| 3 | RNN | 54.31% |
| 4 | RNN | 66.07% |
| 5 | LSTM | 89.57% |
| 6 | LSTM | 90.98% |
| 7 | LSTM | 91.19% |
| 8 | RNN | 56.21% |

---

# 🥇 Best Model

The best random-search trial was:

```text
Trial: 2
Model: LSTM
Validation Accuracy: 91.81%
```

Best parameters:

```python
{
    'model_type': 'lstm',
    'embedding_dim': 64,
    'units': 32,
    'dropout_rate': 0.3,
    'learning_rate': 0.01
}
```

These parameters were selected as the final model configuration.

---

# 🧠 MODULE 5 — Final Sentiment Model

The selected LSTM configuration is used to train the final model.

The final model uses:

```text
Model type      = LSTM
Embedding       = 64
LSTM units      = 32
Dropout         = 0.3
Learning rate   = 0.01
Batch size      = 128
```

---

# 💾 Model Checkpoint

The project uses:

```python
ModelCheckpoint(
    'best_amazon_sentiment.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
```

The best model is therefore saved as:

```text
best_amazon_sentiment.h5
```

The notebook also uses:

```python
EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

This prevents unnecessary training when validation loss stops improving.

---

# 📦 Saving the Tokenizer

The tokenizer is saved using Pickle:

```python
with open(
    'best_amazon_sentiment.pkl',
    'wb'
) as f:
    pickle.dump(tokenizer, f)
```

The tokenizer file is:

```text
best_amazon_sentiment.pkl
```

Both the model and tokenizer are required for consistent inference.

---

# 📥 Loading the Model

The saved model is loaded with:

```python
from tensorflow.keras.models import load_model

loaded_model = load_model(
    'best_amazon_sentiment.h5'
)
```

The tokenizer is loaded:

```python
with open(
    'best_amazon_sentiment.pkl',
    'rb'
) as f:

    loaded_tokenizer = pickle.load(f)
```

---

# 📊 Final Sentiment Model Result

The recorded final test accuracy was:

```text
0.9167149169587079
```

Approximately:

# **91.67% Test Accuracy**

The model therefore correctly classified approximately 91.67% of the test reviews in the recorded experiment.

---

# 🔮 Testing Sentiment Prediction

The project defines:

```python
def predict_sentiment(text):
```

The process is:

```text
Input Text
    ↓
Clean Text
    ↓
Tokenizer
    ↓
Sequence
    ↓
Padding
    ↓
Loaded LSTM
    ↓
Probability
    ↓
Positive / Negative
```

Example:

```python
predict_sentiment(
    "The product was excellent"
)
```

The output is a sentiment label and probability.

The notebook recorded an example prediction:

```text
Positive
Probability ≈ 0.6611
```

---

# 🌐 MODULE 6 — English to Hindi Seq2Seq Translation

The next section implements a basic **Encoder-Decoder Sequence-to-Sequence model**.

The project uses a small manually defined dataset containing English-Hindi sentence pairs.

Example:

```text
English:
i am happy

Hindi:
मैं खुश हूँ
```

---

# Translation Dataset

The notebook contains the following pairs:

```python
data = [
    ("i am happy", "मैं खुश हूँ"),
    ("You are sad", "आप दुखी हैं"),
    ("she is tired", "वह थक गया है"),
    ("we are hungry", "हम भूखें है"),
    ("they are busy", "वे व्यस्त हैं"),
    ("i am cold", "मुझे ठंड लग रही है"),
    ("you are late", "तुम देरी से आए हो"),
    ("she is happy", "वह खुश है"),
    ("we are ready", "हम तैयार हैं")
]
```

This is a very small educational dataset.

It is intended to demonstrate the architecture rather than provide production-quality translation.

---

# 📚 Vocabulary Creation

A custom vocabulary builder is implemented:

```python
def build_vocab(sentences, lang):
```

Special tokens are:

```text
<PAD> = 0
<SOS> = 1
<EOS> = 2
```

The project builds:

```python
eng_vocab
hin_vocab
```

The recorded vocabulary sizes are:

```text
English Vocabulary Size = 20
Hindi Vocabulary Size   = 27
```

---

# 🔢 Sequence Preparation

Each sentence is converted into integer indices.

The function:

```python
sentence_to_indices()
```

adds:

```text
<SOS>
```

at the beginning and:

```text
<EOS>
```

at the end.

Example:

```text
Sentence:

i am happy

↓

<SOS> i am happy <EOS>
```

Then the sequences are padded using:

```python
tf.keras.preprocessing.sequence.pad_sequences()
```

---

# 🧠 Encoder

The Encoder is implemented using:

```python
class Encoder(tf.keras.Model)
```

Architecture:

```text
Input Tokens
     ↓
Embedding
     ↓
LSTM
     ↓
Hidden State
Cell State
```

The encoder contains:

```python
self.embedding = tf.keras.layers.Embedding(
    input_size,
    embed_size
)

self.lstm = tf.keras.layers.LSTM(
    hidden_size,
    return_state=True
)
```

---

# 🧠 Decoder

The Decoder uses:

```text
Embedding
   ↓
LSTM
   ↓
Dense
   ↓
Vocabulary Prediction
```

The decoder contains:

```python
self.embedding = tf.keras.layers.Embedding(
    output_size,
    embed_size
)

self.lstm = tf.keras.layers.LSTM(
    hidden_size,
    return_state=True,
    return_sequences=True
)

self.fc = tf.keras.layers.Dense(
    output_size
)
```

---

# 🔗 Seq2Seq Model

The Encoder and Decoder are combined:

```python
class Seq2Seq(tf.keras.Model)
```

The model uses:

```text
Encoder
   ↓
Hidden State + Cell State
   ↓
Decoder
   ↓
Hindi Tokens
```

The implementation also includes teacher forcing during training.

---

# 🏋️ Seq2Seq Training

The model configuration is:

```text
English vocabulary = 20
Hindi vocabulary = 27
Embedding size = 50
LSTM hidden size = 100
```

The model was trained for:

```text
50 epochs
```

The recorded loss progression was approximately:

```text
Epoch 0  → 3.2962
Epoch 10 → 1.9390
Epoch 20 → 0.7216
Epoch 30 → 0.1596
Epoch 40 → 0.0247
```

The training loss decreased substantially.

---

# 🔤 Translation Function

The notebook defines:

```python
translate(
    model,
    sentence,
    eng_vocab,
    hin_vocab,
    max_len=15
)
```

The function:

1. Tokenizes the input sentence.
2. Converts tokens into indices.
3. Adds `<SOS>` and `<EOS>`.
4. Encodes the sentence.
5. Initializes the decoder.
6. Generates Hindi tokens one at a time.
7. Stops when `<EOS>` is generated.
8. Converts tokens back into words.

---

# 🌍 Translation Results

The model successfully reproduced several sentences from its small training dataset.

### Example 1

```text
Input:
i am happy

Expected:
मैं खुश हूँ

Predicted:
मैं खुश हूँ
```

### Example 2

```text
Input:
You are sad

Expected:
आप दुखी हैं

Predicted:
आप दुखी हैं
```

### Example 3

```text
Input:
we are ready

Expected:
हम तैयार हैं

Predicted:
हम तैयार हैं
```

The project also tests:

```text
tired is hungry
```

The model generated:

```text
वह थक गया है
```

However, this input was not one of the exact dataset pairs, so this should not be interpreted as general translation performance.

---

# ⚠️ Seq2Seq Limitation

This translation model is primarily educational.

It is trained on only:

```text
9 sentence pairs
```

Therefore, it should not be considered a general English-to-Hindi translation system.

A production translation system would require:

- A much larger parallel corpus
- Better tokenization
- Subword tokenization
- Attention
- Transformer architecture
- Proper train/validation/test splits
- BLEU/ROUGE or other evaluation metrics
- More robust handling of unknown words

---

# ⚡ MODULE 7 — Tiny Transformer

The notebook also implements a very small Transformer-style architecture using PyTorch.

The model is:

```python
class TinyTransformer(nn.Module)
```

It contains:

```text
Hindi Embedding
English Embedding
      ↓
Multihead Attention
      ↓
Feed Forward Layer
      ↓
Output Vocabulary
```

The implementation uses:

```python
nn.MultiheadAttention(
    d_model,
    num_heads=1,
    batch_first=True
)
```

The default model dimension is:

```text
d_model = 32
```

The model is intentionally small and educational.

---

# 🧠 MODULE 8 — Sentence Transformers

The project also demonstrates sentence embeddings using:

```python
from sentence_transformers import SentenceTransformer
```

The model used is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Example:

```python
sentences = [
    "This is an example sentence",
    "Each sentence is converted"
]

model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2'
)

embeddings = model.encode(sentences)
```

The result is a numerical vector representation of the sentences.

This is important for:

- Semantic search
- Similarity
- Retrieval
- RAG
- Document search
- Clustering
- Recommendation systems

---

# 🦙 MODULE 9 — Local Llama + PDF RAG

The next major section builds a local **Retrieval-Augmented Generation (RAG)** system.

The workflow is:

```text
PDF
 ↓
PDF Loader
 ↓
Documents
 ↓
Embeddings
 ↓
Vector Store
 ↓
Retriever
 ↓
Relevant Context
 ↓
Prompt
 ↓
Local Llama
 ↓
Answer
```

---

# Installing Llama.cpp

The notebook installs:

```bash
pip install langchain langchain-community pypdf docarray sentence-transformers huggingface_hub llama-cpp-python
```

It also installs build dependencies:

```bash
apt-get update
apt-get install -y git cmake build-essential
```

Then clones Llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp
```

and builds it.

This section was originally designed for Google Colab/Linux.

---

# Downloading the Llama Model

The project downloads:

```text
Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

from:

```text
bartowski/Llama-3.2-3B-Instruct-GGUF
```

using:

```python
from huggingface_hub import hf_hub_download
```

Example:

```python
model_path = hf_hub_download(
    repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
    filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)
```

---

# 📄 Uploading a PDF

The notebook uses Google Colab's upload functionality:

```python
from google.colab import files

uploaded = files.upload()

pdf_path = list(uploaded.keys())[0]
```

Therefore, the user can upload a PDF directly into the Colab environment.

---

# 🦙 Loading Llama

The project uses:

```python
from langchain_community.llms import LlamaCpp
```

The Llama model is initialized using:

```python
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=2048,
    n_gpu_layers=33 if torch.cuda.is_available() else 0,
    temperature=0.7,
    verbose=False
)
```

Configuration:

```text
Context window = 2048
Temperature = 0.7
GPU layers = 33 when CUDA is available
```

---

# 📑 Loading the PDF

The project uses:

```python
from langchain_community.document_loaders import PyPDFLoader
```

The document is loaded:

```python
loader = PyPDFLoader(pdf_path)

pages = loader.load_and_split()
```

The sample PDF used in the notebook is a clinical trial report.

It contains information about:

```text
Clinical Trial: T001
Drug: Atorvastatin
Condition: High Cholesterol
Participants: 150
Trial Duration: 12 weeks
Primary Outcome: Reduction in LDL levels
Safety Events: 5
Efficacy Rate: 78%
```

---

# 🧬 Creating Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

through LangChain:

```python
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
```

Each document chunk is converted into an embedding vector.

---

# 🗄️ Vector Store

The project uses:

```python
DocArrayInMemorySearch
```

The vector store is created:

```python
vectorstore = DocArrayInMemorySearch.from_documents(
    pages,
    embedding=embeddings
)
```

A retriever is then created:

```python
retriever = vectorstore.as_retriever()
```

The retriever searches for relevant document content based on semantic similarity.

---

# 🔗 RAG Pipeline

The RAG chain is built using:

```python
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

The pipeline is:

```text
User Question
      ↓
Retriever
      ↓
Relevant PDF Context
      ↓
Prompt Template
      ↓
Llama
      ↓
String Output
```

---

# 📝 RAG Prompt

The project uses:

```text
Use the following context to answer the question:

{context}

Question:
{question}

Answer:
```

This is a basic RAG prompt.

---

# ❓ Asking Questions

A question can be sent using:

```python
question = "What is the main topic of the PDF?"

response = chain.invoke(question)
```

The user can replace the question with another question related to the uploaded PDF.

---

# ⏱️ Measuring Inference Time

The project also measures inference time:

```python
import time

st = time.time()

response = chain.invoke(question)

ed = time.time()

print(
    f"Inference is at : {(ed-st):.3f} Seconds"
)
```

This can be used to evaluate how long the local RAG pipeline takes to generate an answer.

---

# ⚠️ RAG Example Note

The notebook includes a test question:

```text
how to make pizza
```

This question is unrelated to the clinical-trial PDF.

The returned answer demonstrates that a local LLM can generate text, but it should **not** be treated as evidence that the RAG system correctly answered a document-grounded question.

For proper RAG evaluation, ask questions directly supported by the uploaded PDF, such as:

```text
What drug was investigated in the clinical trial?

How many participants were enrolled?

What was the primary outcome?

What was the efficacy rate?

How many safety events were reported?
```

---

# ✨ MODULE 10 — Google Gemini API

The project also demonstrates direct API usage with Google Gemini.

The notebook installs:

```bash
pip install google-generativeai
```

and uses:

```python
import google.generativeai as genai
```

---

# Gemini Setup

The project initializes:

```python
model = genai.GenerativeModel(
    'gemini-2.0-flash'
)
```

The API key should be provided securely through an environment variable.

Recommended:

```python
import os
import google.generativeai as genai

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)
```

---

# Gemini Example

The notebook sends:

```python
response = model.generate_content(
    "Provide a very short response, what is ai?"
)
```

The recorded response was:

```text
AI is intelligence demonstrated by machines.
```

This demonstrates basic Gemini text generation.

---

# 🌐 MODULE 11 — Tavily Web Search

The project uses Tavily for web search.

Import:

```python
from tavily import TavilyClient
```

Initialize:

```python
tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)
```

A search can be performed:

```python
search_result = tavily.search(
    "latest AI news",
    max_results=2
)
```

The notebook retrieved AI news sources including:

```text
AI News
TechCrunch Artificial Intelligence
```

The result contains:

```text
Title
URL
Content
```

---

# 🤖 MODULE 12 — AI Fact Checker

The project combines:

```text
Tavily
+
Google Gemini
```

to create an AI-powered fact-checking workflow.

The idea is:

```text
User Claim
    ↓
Tavily Search
    ↓
Evidence
    ↓
Gemini
    ↓
Fact-Check Analysis
    ↓
Verdict
Confidence
Explanation
Evidence
Sources
```

---

# 🔎 Fact Checker Workflow

The main function is:

```python
def fact_check(claim):
```

The function:

1. Receives a claim.
2. Searches the web.
3. Collects up to five results.
4. Extracts titles.
5. Extracts content.
6. Extracts URLs.
7. Builds an evidence block.
8. Sends the claim + evidence to Gemini.
9. Generates a verdict.
10. Prints the sources.

---

# 📋 Fact-Checking Categories

The Gemini prompt asks the model to choose one of:

```text
TRUE
FALSE
PARTIALLY TRUE
UNVERIFIED
OUTDATED
```

It also requests:

```text
Confidence
Explanation
Key Evidence
```

---

# 🧪 Fact Checker Example

The notebook tests:

```text
the earth is flat
```

The recorded result was:

```text
VERDICT: FALSE
CONFIDENCE: High
```

The assistant also returned supporting sources.

---

# 💬 Interactive Fact Checker

The project includes:

```python
def main():
```

The application runs interactively:

```text
SIMPLE FACT CHECKER WITH GEMINI AND TAVILY

Enter your claim to fact-check:
```

The user can type a claim.

The program searches the web and sends the collected evidence to Gemini.

To exit:

```text
quit
```

or:

```text
q
```

or:

```text
exit
```

---

# ✈️ MODULE 13 — Travel Planning Assistant

The final major application is a travel-planning assistant.

It combines:

```text
User Input
+
Tavily Web Search
+
Google Gemini
```

The assistant gathers:

- Destination
- Duration
- Budget
- Interests
- Travel style

and creates a personalized travel plan.

---

# 🧳 Travel Assistant Workflow

The architecture is:

```text
User
 ↓
Destination Information
 ↓
Travel Research
 ↓
Tavily Web Search
 ↓
Current Travel Information
 ↓
Gemini
 ↓
Personalized Travel Plan
 ↓
Travel Alerts
 ↓
Optional File Save
```

---

# 📝 Travel Information Collection

The assistant asks:

```text
Where do you want to go?

How many days?

What's your budget range?

What are you interested in?
```

Possible interests include:

```text
food
culture
adventure
nature
shopping
history
```

---

# 🎒 Travel Styles

The project provides:

```text
1. Backpacker/Budget
2. Comfort/Mid-range
3. Luxury
4. Adventure
5. Cultural/Historical
```

The selected value is stored in:

```python
self.travel_style
```

---

# 🔎 Travel Research

The assistant creates several search queries.

Examples:

```text
{destination} travel guide 2025-2026

{destination} best attractions things to do

{destination} travel safety current situation

{destination} weather climate best time to visit

{destination} budget costs accommodation and food
```

Each query is sent to Tavily.

The assistant collects up to three search results per query.

---

# 🧠 Travel Plan Generation

The collected research is combined and passed to Gemini.

The prompt requests:

## 1. Destination Overview

Including:

- Brief description
- Highlights
- Best time to visit
- Cultural tips
- Etiquette

## 2. Safety & Practical Information

Including:

- Current safety situation
- Visa requirements
- Currency
- Payment methods
- Language tips

## 3. Daily Itinerary

Including:

- Day-by-day activities
- Attractions
- User interests
- Travel time between locations

## 4. Food

Including:

- Local cuisine
- Vegetarian information
- Non-vegetarian information
- Food information for vegetarian/vegan travelers

## 5. Emergency Information

Including:

- Important phone numbers
- Embassy/consulate information

---

# ⚠️ Travel Alerts

The assistant also performs a travel advisory search.

The intended query is:

```text
{destination} travel advisory warning alert 2025 2026
```

The results are displayed as current travel alerts.

Because travel information changes over time, users should verify important safety, visa, health, and emergency information through official sources before relying on the generated itinerary.

---

# 💾 Saving the Travel Plan

The application asks:

```text
Save this plan to file? (y/n)
```

If the user chooses:

```text
y
```

the travel plan and alerts are written to a text file.

---

# ▶️ How to Run the Complete Project

The notebook contains multiple independent sections.

You do **not** have to run every section if you only want one specific application.

Recommended execution order:

```text
1. Install dependencies
2. Run Amazon sentiment analysis
3. Train RNN
4. Train LSTM
5. Run hyperparameter search
6. Train final model
7. Save/load sentiment model
8. Run sentiment predictions
9. Run Seq2Seq translation
10. Run Tiny Transformer
11. Generate sentence embeddings
12. Set up local Llama
13. Upload PDF
14. Build RAG
15. Configure Gemini
16. Configure Tavily
17. Run fact checker
18. Run travel assistant
```

---

# ☁️ Running in Google Colab

Google Colab is the easiest environment for reproducing the original notebook.

## Step 1

Open Google Colab.

## Step 2

Upload:

```text
Gen_ai_ff_final.ipynb
```

## Step 3

Run the installation cells.

## Step 4

Run the Amazon review section.

The dataset will be downloaded using:

```python
kagglehub.dataset_download(
    "snap/amazon-fine-food-reviews"
)
```

## Step 5

For the PDF RAG section, upload your PDF when prompted.

## Step 6

Set your API keys securely.

Do not paste real API keys into cells that will be committed to GitHub.

---

# 🖥️ Running Locally

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Move into the repository:

```bash
cd GenAI-NLP-Project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

Activate on Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
Gen_ai_ff_final.ipynb
```

---

# 🧾 Recommended requirements.txt

A starting `requirements.txt` can contain:

```text
numpy
pandas
matplotlib
scikit-learn
tensorflow
torch
kagglehub
sentence-transformers
transformers
huggingface-hub
langchain
langchain-community
pypdf
docarray
llama-cpp-python
google-generativeai
tavily-python
python-dotenv
jupyter
```

For the Llama.cpp section, additional system/build dependencies may be required depending on the operating system.

---

# 🗂️ Recommended .gitignore

Create:

```text
.gitignore
```

with:

```text
.env
venv/
__pycache__/
.ipynb_checkpoints/
*.pyc

best_amazon_sentiment.h5
best_amazon_sentiment.pkl

*.gguf
```

Large model files should generally not be committed directly to a normal GitHub repository.

---

# 📊 Expected Outputs

The major recorded results are summarized below.

## Amazon Dataset

```text
Total reviews = 568,454
```

## Rating Distribution

```text
1 star = 52,268
2 star = 29,769
3 star = 42,640
4 star = 80,655
5 star = 363,122
```

## Binary Dataset

```text
Negative = 82,037
Positive = 443,777
```

## Balanced Dataset

```text
Negative = 82,037
Positive = 82,037

Total = 164,074
```

## RNN

```text
Final training accuracy ≈ 67.33%
```

## LSTM

```text
Final training accuracy ≈ 93.22%
```

## Best Hyperparameter Search Trial

```text
Model = LSTM

Embedding dimension = 64
Units = 32
Dropout = 0.3
Learning rate = 0.01

Validation accuracy ≈ 91.81%
```

## Final Sentiment Model

```text
Test Accuracy ≈ 91.67%
```

## Seq2Seq

The model successfully generated several training examples such as:

```text
i am happy
→
मैं खुश हूँ
```

## Gemini

Example:

```text
Question:
What is AI?

Response:
AI is intelligence demonstrated by machines.
```

---

# 🧠 Important Implementation Notes

## 1. The project is a collection of experiments

This notebook is not a single production application.

It is a learning/project notebook containing several AI implementations.

---

## 2. Some sections are independent

For example:

```text
Sentiment Analysis
```

does not depend on:

```text
Travel Buddy
```

Similarly:

```text
Seq2Seq
```

is independent from:

```text
PDF RAG
```

This makes each section useful as an individual learning module.

---

## 3. GPU

GPU acceleration is particularly useful for:

- TensorFlow RNN/LSTM training
- PyTorch models
- Llama.cpp
- Large embedding models

CPU execution is possible for many sections but may be slower.

---

## 4. Large Models

The Llama section downloads:

```text
Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

The model is approximately several GB in size.

Make sure your machine has enough:

- Disk space
- RAM
- GPU VRAM if GPU acceleration is used

---

## 5. Sentence Transformer Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model will be downloaded when first used if it is not already cached.

---

# ⚠️ Known Issues and Limitations

## 1. Hard-Coded API Keys

The original notebook contains API keys directly in code.

These must be removed before publishing.

Use environment variables instead.

---

## 2. Deprecated LangChain API

The notebook uses:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
```

The notebook itself produces a deprecation warning indicating that the embedding implementation should be migrated to the newer LangChain Hugging Face package.

For a refreshed implementation, consider:

```bash
pip install -U langchain-huggingface
```

and:

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

---

## 3. HDF5 Model Format

The final sentiment model is saved as:

```text
best_amazon_sentiment.h5
```

The notebook reports that HDF5 is considered a legacy model-saving format in current Keras versions.

A modern implementation can use:

```text
.keras
```

instead.

---

## 4. RNN Performance

The basic RNN achieved much lower validation accuracy than the LSTM.

This demonstrates the advantage of using LSTM-style recurrent memory for this particular experiment.

---

## 5. Small Translation Dataset

The English-Hindi Seq2Seq model uses only nine sentence pairs.

It is therefore not a general translation model.

---

## 6. Tiny Transformer

The Transformer implementation is intentionally minimal.

It is primarily intended for understanding:

- Embeddings
- Attention
- Feed-forward layers

rather than production translation.

---

## 7. RAG Evaluation

The PDF RAG section should be evaluated using questions that are actually answerable from the PDF.

Testing it with unrelated questions can cause the LLM to generate general knowledge instead of document-grounded information.

---

## 8. Travel Information

Travel information is time-sensitive.

The generated itinerary should not be treated as authoritative.

Always verify:

- Visa requirements
- Entry requirements
- Travel advisories
- Emergency numbers
- Local laws
- Health requirements
- Transport schedules

with official sources before traveling.

---

# 🛠️ Recommended Improvements

The project can be improved significantly.

---

## Sentiment Analysis Improvements

### Use pretrained Transformers

Instead of only RNN/LSTM, experiment with:

```text
BERT
RoBERTa
DistilBERT
ALBERT
```

---

## Better Text Preprocessing

Consider:

```text
Lemmatization
Stopword handling
Subword tokenization
Emoji processing
Negation handling
Spelling normalization
```

---

## Better Evaluation

Add:

```text
Confusion Matrix
Precision
Recall
F1 Score
ROC-AUC
PR-AUC
```

Example:

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

---

## Cross Validation

Use:

```text
Stratified K-Fold
```

for more reliable evaluation.

---

## Model Tuning

The existing random search could be extended with:

```text
Optuna
KerasTuner
RandomizedSearchCV
Bayesian Optimization
```

---

# 🚀 Advanced RAG Improvements

The current RAG system can be improved by adding:

```text
Recursive Character Text Splitter
Better chunk sizes
Chunk overlap
Metadata filtering
Top-k retrieval
Similarity threshold
Reranking
Hybrid search
FAISS
Chroma
Qdrant
Pinecone
```

A stronger architecture would be:

```text
PDF
 ↓
Document Loader
 ↓
Text Splitter
 ↓
Embeddings
 ↓
Vector Database
 ↓
Retriever
 ↓
Reranker
 ↓
Prompt
 ↓
LLM
 ↓
Grounded Answer
```

---

# 🧠 Advanced LLM Improvements

The local Llama application could be extended with:

- Streaming responses
- Conversation memory
- Chat history
- System prompts
- Structured outputs
- Function calling
- Tool use
- Agent workflows
- Guardrails
- Evaluation
- Citation generation

---

# 🤖 Fact Checker Improvements

The fact checker can be improved by:

1. Searching multiple independent sources.
2. Ranking source quality.
3. Separating primary and secondary sources.
4. Adding source citations.
5. Checking publication dates.
6. Detecting contradictory evidence.
7. Using structured JSON output.
8. Adding confidence calibration.
9. Recording the evidence used.
10. Adding human review for high-risk claims.

A better workflow would be:

```text
Claim
 ↓
Query Generation
 ↓
Web Search
 ↓
Source Filtering
 ↓
Evidence Extraction
 ↓
Contradiction Detection
 ↓
LLM Analysis
 ↓
Structured Verdict
 ↓
Sources
```

---

# ✈️ Travel Assistant Improvements

The travel assistant could be upgraded with:

```text
Google Maps
Flight APIs
Hotel APIs
Weather APIs
Currency APIs
Local attraction APIs
Restaurant APIs
Real-time transport data
```

It could also produce:

```text
PDF itinerary
Budget breakdown
Maps
Daily schedules
Restaurant recommendations
Hotel recommendations
Packing checklist
Travel checklist
Emergency information
```

---

# 🔒 Security Best Practices

For anyone deploying this project:

### Never store API keys in:

```text
.ipynb
.py
README.md
GitHub
requirements.txt
```

### Use:

```text
.env
Environment Variables
Secret Managers
GitHub Secrets
Cloud Secret Managers
```

### Never commit:

```text
.env
*.key
*.pem
API tokens
Access tokens
Passwords
Private credentials
```

---

# 🧪 Reproducibility

For reproducible results:

```text
Python version
TensorFlow version
PyTorch version
Scikit-learn version
Random seeds
Dataset version
Model version
```

should be recorded.

The notebook uses several explicit seeds, including:

```text
random_state = 21
random_state = 42
```

for different parts of the workflow.

However, deep-learning results can still vary across:

- Hardware
- TensorFlow versions
- CUDA versions
- GPU implementations
- Random initialization

---

# 📚 Learning Outcomes

After completing this project, a learner should understand the basic workflow behind:

## NLP

```text
Text
 ↓
Cleaning
 ↓
Tokenization
 ↓
Numerical Representation
 ↓
Sequence Modeling
```

## RNN

```text
Sequence
 ↓
Hidden State
 ↓
Prediction
```

## LSTM

```text
Sequence
 ↓
Memory Cells
 ↓
Hidden State
 ↓
Prediction
```

## Seq2Seq

```text
Source Sentence
 ↓
Encoder
 ↓
Context
 ↓
Decoder
 ↓
Target Sentence
```

## Transformer

```text
Input
 ↓
Embedding
 ↓
Attention
 ↓
Feed Forward
 ↓
Output
```

## RAG

```text
Documents
 ↓
Chunks
 ↓
Embeddings
 ↓
Vector Store
 ↓
Retriever
 ↓
LLM
 ↓
Answer
```

## AI Agent

```text
User
 ↓
LLM
 ↓
Tools
 ↓
Web Search
 ↓
Reasoning
 ↓
Response
```

---

# 📊 End-to-End Skills Demonstrated

This project demonstrates hands-on experience with:

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras
- PyTorch
- NLP
- Text preprocessing
- Tokenization
- Sequence modeling
- RNN
- LSTM
- Hyperparameter tuning
- Model checkpointing
- Model persistence
- Pickle
- Encoder-Decoder
- Seq2Seq
- Transformer
- Attention
- Sentence embeddings
- Hugging Face
- Sentence Transformers
- Llama
- Llama.cpp
- LangChain
- RAG
- Vector stores
- PDF processing
- Gemini
- Tavily
- Web search
- Fact checking
- AI agents
- Prompt engineering
- Generative AI

---

# 🏆 Project Highlights

## Sentiment Analysis

```text
Dataset:
568,454 Amazon reviews
```

Final model:

```text
LSTM
```

Recorded test accuracy:

```text
91.67%
```

---

## Hyperparameter Search

Best configuration:

```text
Model       = LSTM
Embedding   = 64
Units       = 32
Dropout     = 0.3
Learning    = 0.01
```

Validation accuracy:

```text
91.81%
```

---

## Seq2Seq Translation

Demonstrated:

```text
English → Hindi
```

with a custom Encoder-Decoder LSTM architecture.

---

## Local LLM

Demonstrated:

```text
Llama 3.2 3B Instruct GGUF
```

using Llama.cpp.

---

## RAG

Demonstrated:

```text
PDF → Embeddings → Vector Store → Retriever → Llama
```

---

## Generative AI APIs

Demonstrated:

```text
Google Gemini
Tavily
```

---

## AI Applications

Built:

```text
AI Fact Checker
AI Travel Planning Assistant
```

---

# 🔮 Future Improvements

Possible future versions of this project can include:

## Version 2

```text
Transformer-based sentiment analysis
```

## Version 3

```text
BERT / RoBERTa sentiment classification
```

## Version 4

```text
Production RAG application
```

## Version 5

```text
Agentic RAG
```

## Version 6

```text
Multi-agent travel planner
```

## Version 7

```text
Streamlit / Gradio Web Application
```

## Version 8

```text
FastAPI Backend
+
React Frontend
```

---

# 🌐 Potential Application Architecture

A production version could be structured as:

```text
                    Frontend
                       |
                       ↓
                    FastAPI
                       |
          +------------+-------------+
          |            |             |
          ↓            ↓             ↓
     Sentiment       RAG         AI Agents
       Model         System          |
          |            |             |
          ↓            ↓             ↓
       LSTM        Vector DB      Tavily
                       |          Gemini
                       ↓
                     Llama
```

---

# 📌 Important Reproduction Notes

The original notebook was developed primarily in a **Google Colab environment**.

Some commands are Colab/Linux-specific, such as:

```python
from google.colab import files
```

and shell commands such as:

```bash
!apt-get update
```

These commands will not work unchanged on standard Windows Python environments.

For local execution, these parts must be adapted.

---

# 🖥️ Running Individual Modules

You do not need to run the entire notebook every time.

## If you only want sentiment analysis

Run:

```text
Cells 0–50
```

---

## If you only want Seq2Seq

Run:

```text
Cells 51–73
```

---

## If you only want the Transformer experiment

Run:

```text
Cells 74–78
```

---

## If you only want PDF RAG

Run:

```text
Cells 79–94
```

---

## If you only want Gemini/Tavily

Run:

```text
Cells 97–109
```

---

## If you only want the Travel Assistant

Run the Gemini/Tavily setup first, then:

```text
Cells 110–112
```

---

# 🧭 Recommended Learning Order

If you are studying this project, follow this order:

```text
1. Understand the Amazon dataset
        ↓
2. Learn text preprocessing
        ↓
3. Understand tokenization
        ↓
4. Understand embeddings
        ↓
5. Train RNN
        ↓
6. Train LSTM
        ↓
7. Perform hyperparameter search
        ↓
8. Save/load model
        ↓
9. Understand Encoder-Decoder
        ↓
10. Build Seq2Seq
        ↓
11. Understand attention
        ↓
12. Understand Transformer
        ↓
13. Learn sentence embeddings
        ↓
14. Learn LLM inference
        ↓
15. Learn RAG
        ↓
16. Learn web search tools
        ↓
17. Build AI fact checker
        ↓
18. Build AI travel assistant
```

---

# 🧾 Project Summary

This project demonstrates a progression from traditional NLP and deep learning to modern Generative AI systems.

The first stage focuses on **Amazon customer review sentiment analysis**, where 568,454 reviews are analyzed and converted into a balanced binary classification dataset.

RNN and LSTM models are developed and compared. Hyperparameter search is then used to identify a better LSTM configuration. The final sentiment model achieves a recorded **91.67% test accuracy**.

The project then moves into **sequence-to-sequence learning**, implementing an Encoder-Decoder LSTM for English-to-Hindi translation.

A small Transformer implementation is subsequently explored to demonstrate attention-based modeling.

The project then progresses to **sentence embeddings and Large Language Models**, using Sentence Transformers and a local Llama 3.2 3B model.

A **Retrieval-Augmented Generation system** is built using PDF documents, Hugging Face embeddings, a vector store, LangChain, and Llama.cpp.

Finally, the project demonstrates API-based Generative AI applications using:

```text
Google Gemini
Tavily
```

These tools are combined to create:

```text
AI Fact Checker
```

and:

```text
AI Travel Planning Assistant
```

Overall, the project provides a practical progression:

```text
NLP
 ↓
Deep Learning
 ↓
RNN
 ↓
LSTM
 ↓
Seq2Seq
 ↓
Transformer
 ↓
Embeddings
 ↓
LLM
 ↓
RAG
 ↓
Web Search
 ↓
AI Agents
```

---

# 👨‍💻 Author

**Rakshith S**

Data Science | Machine Learning | Generative AI

---

# ⭐ If You Find This Project Useful

If you find this project useful for learning NLP, Deep Learning, Generative AI, LLMs, or RAG, consider giving the repository a ⭐ on GitHub.

---

# 📜 License

This project can be released under the MIT License.

Example:

```text
MIT License

Copyright (c) 2026 Rakshith S

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, subject to the conditions of the MIT License.
```

---

# ⚠️ Disclaimer

This repository is primarily an educational and experimental project.

The outputs generated by:

- Large Language Models
- Web search
- Fact-checking systems
- Travel planning systems
- Machine translation systems

should not automatically be treated as authoritative.

Always independently verify important information, particularly:

- Medical information
- Financial information
- Legal information
- Travel safety information
- Visa requirements
- Emergency information
- Scientific claims

The AI systems demonstrated in this project are intended to show implementation concepts and should be validated before being used in production or high-risk applications.

---

# 🎯 Final Project Snapshot

```text
┌───────────────────────────────────────────────┐
│        GENERATIVE AI & NLP PROJECT            │
├───────────────────────────────────────────────┤
│                                               │
│  Amazon Fine Food Reviews                     │
│          ↓                                    │
│  Sentiment Analysis                           │
│          ↓                                    │
│  RNN → LSTM → Hyperparameter Search           │
│          ↓                                    │
│  Final Accuracy: ~91.67%                      │
│          ↓                                    │
│  English → Hindi Seq2Seq                      │
│          ↓                                    │
│  Tiny Transformer                             │
│          ↓                                    │
│  Sentence Transformers                        │
│          ↓                                    │
│  Local Llama 3.2                              │
│          ↓                                    │
│  PDF RAG                                      │
│          ↓                                    │
│  Google Gemini                                │
│          +                                    │
│  Tavily Web Search                            │
│          ↓                                    │
│  AI Fact Checker                              │
│          ↓                                    │
│  AI Travel Planning Assistant                 │
│                                               │
└───────────────────────────────────────────────┘
```

---

# 🚀 End-to-End AI Journey

The most important takeaway from this project is the progression from **building models** to **building AI systems**.

```text
Model Building
      ↓
Deep Learning
      ↓
NLP
      ↓
Generative Models
      ↓
Large Language Models
      ↓
Retrieval-Augmented Generation
      ↓
External Tools
      ↓
AI Applications
      ↓
Agentic AI
```

This project therefore serves as a practical foundation for further work in:

```text
Data Science
Machine Learning
Deep Learning
Natural Language Processing
Generative AI
Large Language Models
RAG
Agentic AI
AI Applications
```
