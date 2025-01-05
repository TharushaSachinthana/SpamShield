import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer

def load_models():
    classifier = pickle.load(open('model/classifier.pkl', 'rb'))
    index = faiss.read_index('model/faiss_index')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return classifier, index, model

def classify_email(email_text, classifier, index, model, k=5):
    query_embedding = np.array([model.encode(email_text)]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    return indices

if __name__ == "__main__":
    email_text = "Claim your free gift card now!"
    classifier, index, model = load_models()
    result = classify_email(email_text, classifier, index, model)
    print(f"Classification result: {result}")
