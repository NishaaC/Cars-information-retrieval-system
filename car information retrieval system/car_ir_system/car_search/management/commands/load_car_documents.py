import os
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words and stem
    tokens = [ps.stem(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def load_documents(folder_path):
    documents = []
    file_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(preprocess_text(content))
                file_names.append(filename)
    return documents, file_names

def create_tfidf_matrix(documents):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(documents)

def search(query, tfidf_matrix, documents, file_names):
    query_vector = TfidfVectorizer().fit(documents).transform([preprocess_text(query)])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]
    
    results = []
    for idx in related_docs_indices[:5]:  # Top 5 results
        results.append({
            'filename': file_names[idx],
            'similarity': cosine_similarities[idx],
            'content': documents[idx][:200] + '...'  # Preview of content
        })
    
    return results