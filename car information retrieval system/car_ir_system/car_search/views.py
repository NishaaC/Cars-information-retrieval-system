from django.shortcuts import render
from .ir_system import load_documents, create_tfidf_matrix, search
import os
import nltk
import pandas
nltk.download('punkt')


# Load documents and create TF-IDF matrix (do this once when server starts)
folder_path = 'C:/Users/nisha/OneDrive/Documents/car information retrieval system/car_ir_system/cars'


documents, file_names = load_documents(folder_path)
tfidf_matrix = create_tfidf_matrix(documents)

def home(request):
    return render(request, 'car_search/home.html')

def search_results(request):
    query = request.GET.get('query', '')
    if query:
        results = search(query, tfidf_matrix, documents, file_names)
    else:
        results = []
    return render(request, 'car_search/search_results.html', {'query': query, 'results': results})

def car_detail(request, filename):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return render(request, 'car_search/car_detail.html', {'filename': filename, 'content': content})