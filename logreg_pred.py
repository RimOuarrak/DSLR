import csv
import sys
import numpy as np

# Mapping numbers back to house names
house_mapping = {0:"Gryffindor", 1:"Hufflepuff", 2:"Ravenclaw", 3:"Slytherin"}

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def preprocess(X):
    """Normalize and add bias (same as training)"""
    X = np.array(X, dtype=float)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min + 1e-8)
    bias = np.ones((X.shape[0], 1))
    X = np.hstack((bias, X))
    return X

if len(sys.argv) != 3:
    print("Usage: python logreg_predict.py dataset_test.csv weights.csv")
    sys.exit(1)

dataset_file = sys.argv[1]
weights_file = sys.argv[2]

# Load weights
weights = np.loadtxt(weights_file, delimiter=",")

# Read test dataset
skip_cols = ["Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"]

X = []
indices = []

with open(dataset_file, "r") as f:
    reader = csv.DictReader(f)
    feature_names = [col for col in reader.fieldnames if col not in skip_cols]

    for idx, row in enumerate(reader):
        try:
            features = [float(row[f]) for f in feature_names]
            X.append(features)
            indices.append(idx)
        except (ValueError, TypeError):
            continue

X = preprocess(X)

# Predict probabilities
probs = np.zeros((X.shape[0], weights.shape[0]))
for class_idx in range(weights.shape[0]):
    z = np.dot(X, weights[class_idx])
    probs[:, class_idx] = sigmoid(z)

# Pick class with highest probability
predicted_classes = np.argmax(probs, axis=1)

# Write predictions
with open("houses.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Index", "Hogwarts House"])
    for i, cls in zip(indices, predicted_classes):
        writer.writerow([i, house_mapping[cls]])

print("Predictions saved to houses.csv")
