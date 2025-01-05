import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(embeddings, labels, model_path='model/classifier.pkl'):
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    
    predictions = classifier.predict(embeddings)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    embeddings = np.load('data/test_embeddings.npy')
    labels = np.loadtxt('data/test_labels.csv', delimiter=',').astype(int)
    evaluate_model(embeddings, labels)
