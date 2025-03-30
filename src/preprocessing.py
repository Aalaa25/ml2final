import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Lowercase, remove numbers, punctuation, and stopwords."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_dataframe(df):
    """Preprocess text column for both datasets."""
    df = df.dropna(subset=['text']).reset_index(drop=True)
    df['clean_text'] = df['text'].apply(preprocess_text)
    return df
