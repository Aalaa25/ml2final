from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(text_data):
    """Convert text data into TF-IDF features."""
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix, vectorizer