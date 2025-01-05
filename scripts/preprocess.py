import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss

def load_and_clean_data(dataset_path):
    data = pd.read_csv(dataset_path)
    data['cleaned_text'] = data['text'].apply(
        lambda x: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', x.lower()))
    )
    return data

def generate_embeddings(data, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    data['embeddings'] = data['cleaned_text'].apply(lambda x: model.encode(x))
    embeddings = np.array(data['embeddings'].tolist()).astype('float32')
    return embeddings, model

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    dataset_path = 'data/spam_ham_dataset.csv'
    data = load_and_clean_data(dataset_path)
    embeddings, model = generate_embeddings(data)
    index = create_faiss_index(embeddings)
    
    # Save embeddings and index
    np.save('data/embeddings.npy', embeddings)
    faiss.write_index(index, 'model/faiss_index')
    print("Preprocessing completed. Data, embeddings, and FAISS index saved.")
