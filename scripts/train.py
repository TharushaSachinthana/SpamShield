import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

def train_model(embeddings, labels, save_path='model/classifier.pkl'):
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    
    # Save model
    with open(save_path, 'wb') as f:
        pickle.dump(classifier, f)
    print("Training completed. Model saved.")

if __name__ == "__main__":
    embeddings = np.load('data/embeddings.npy')
    data = np.loadtxt('data/labels.csv', delimiter=',')  # Assuming labels saved separately
    labels = data.astype(int)
    
    train_model(embeddings, labels)
