# Document Clustering Project

## 📌 Overview

This project applies **unsupervised learning** techniques to cluster documents from two datasets:

- **People Wikipedia Dataset** (biographical articles from Wikipedia)
- **20 Newsgroups Dataset** (20,000 articles across different topics)

The goal is to uncover **natural groupings** within the text data.

## 📂 Project Structure

```
├── data/                       # Folder for datasets (raw and preprocessed)
├── src/
│   ├── preprocessing.py        # Data cleaning and preprocessing
│   ├── feature_extraction.py   # TF-IDF & Word Embeddings
│   ├── clustering.py           # K-Means, Hierarchical, GMM, LDA
│   ├── evaluation.py           # Silhouette & Purity Scores
│   ├── visualization.py        # Cluster visualizations
│   ├── main.py                 # Main script to run everything
├── notebooks/                  # Jupyter notebooks for EDA
├── results/                    # Cluster results & plots
├── requirements.txt            # List of required libraries
├── README.md                   # Project documentation
```

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Main Pipeline

```bash
python main.py
```

## 📊 Clustering Methods Used

- **K-Means Clustering**

## 📈 Evaluation Metrics

- **Silhouette Score**

## 🎨 Visualizations

- **PCA** for dimensionality reduction