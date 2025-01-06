# Email Spam Filtering with Retrieval-Augmented Generation (RAG)

## Overview
This repository showcases an advanced **email spam detection system** that leverages **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)** for precise and context-aware email classification. The solution integrates cutting-edge technologies such as **FAISS**, **SentenceTransformer**, and **Gemini LLM**, providing a scalable and real-time classification pipeline with API capabilities.  

This project was developed with the goal of enhancing traditional spam filtering approaches by incorporating contextual understanding and real-time retrieval.

---

## Architecture
The architecture of the system is designed with modularity and scalability in mind:

1. **Data Preprocessing**
   - Raw emails are processed for cleaning, tokenization, and embedding generation using **SentenceTransformer**.

2. **Embedding and Retrieval**
   - A **FAISS** index is built using preprocessed embeddings for fast similarity search and retrieval of relevant email contexts.

3. **Classification and Augmentation**
   - **Gemini LLM** enriches retrieved contexts, enabling better decision-making.
   - A logistic regression model is used for binary classification (`ham` or `spam`).

4. **API Integration**
   - The classification pipeline is exposed via a RESTful API for real-time email classification.

5. **Evaluation**
   - Metrics such as accuracy, precision, recall, and F1 score are computed on real-world datasets to validate the model.

---

## Features
- **RAG-Based Classification**: Combines retrieval and LLM-enhanced generation for accurate and context-aware email spam detection.
- **Real-Time Processing**: Built for high-speed, real-time classification with API integration.
- **Embeddings with SentenceTransformer**: Efficient email embedding generation for semantic similarity.
- **Fast Indexing and Retrieval with FAISS**: Ensures quick context retrieval for enhanced classification.
- **Scalable Architecture**: Suitable for both small-scale and enterprise-level deployment.
- **Comprehensive Evaluation**: Includes detailed metrics reporting for transparent validation.

---

## Tools and Technologies
### Core Technologies
- **Python**: Primary programming language for the project.
- **FAISS**: For building and querying high-speed similarity indices.
- **SentenceTransformer**: Generates embeddings for semantic understanding.
- **Gemini LLM**: Provides contextual augmentation for informed classification.
- **scikit-learn**: Implements logistic regression and evaluation metrics.

### Supporting Tools
- **Flask/FastAPI**: API framework for deploying the model as a service.
- **Pandas & NumPy**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.

---

## Project Structure
```plaintext
email-spam-filtering/
│
├── data/
│   ├── raw/                  # Original datasets (e.g., emails for training/testing).
│   ├── processed/            # Preprocessed datasets.
│   ├── samples/              # Example emails for testing.
│   └── README.md             # Description of the data folder.
│
├── models/
│   ├── faiss_index/          # FAISS index for fast retrieval.
│   ├── trained_model.pth     # Trained logistic regression model.
│   └── README.md             # Notes on model training and loading.
│
├── notebooks/
│   ├── email_spam_filtering.ipynb  # Main implementation notebook.
│   ├── exploratory_data_analysis.ipynb  # EDA and visualizations.
│   └── README.md             # Description of notebooks.
│
├── results/
│   ├── logs/                 # Training and evaluation logs.
│   ├── figures/              # Plots and graphs.
│   ├── metrics.json          # Evaluation metrics (e.g., accuracy, precision).
│   └── README.md             # Summary of results.
│
├── scripts/
│   ├── preprocess.py         # Data preprocessing script.
│   ├── train.py              # Model training script.
│   ├── predict.py            
│   ├── run_pipeline.py       
│   ├── evaluate.py           # Model evaluation script.
│   └── README.md             # Description of scripts.
│
├── tests/
│   ├── test_preprocessing.py # Unit tests for preprocessing functions.
│   ├── test_model.py         # Unit tests for model-related code.
│   └── README.md             # Notes on test coverage.
│
├── .gitignore                # Ignore unnecessary files.
├── LICENSE                   # Project license (e.g., MIT).
├── README.md                 # Project overview (this file).
├── requirements.txt          # Python dependencies.
└── setup.py                  # Install the project as a package.
```

---

## Installation
### Prerequisites
- Python 3.8 or higher
- Pip or Conda for package management

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/TharushaSachinthana/SpamShield.git
   cd SpamShield
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the data:
   - Place raw datasets in `data/raw/` and run `scripts/preprocess.py`.

4. Train the model:
   ```bash
   python scripts/train.py
   ```

5. Test the system:
   ```bash
   python scripts/evaluate.py
   ```

---

## Usage
- **Classification API**: Deploy the model as an API using Flask or FastAPI.
- **Testing on Samples**: Use sample emails in `data/samples/` to validate the model's predictions.
- **Reproducibility**: Jupyter notebooks in `notebooks/` ensure end-to-end reproducibility.

---

## Results
- **Accuracy**: 96.5%
- **Precision**: 94.2%
- **Recall**: 92.8%
- **F1-Score**: 93.5%

Plots and detailed metrics are available in the `results/` folder.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

