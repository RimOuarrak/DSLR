from logreg_class import LogisticRegression
import csv
import sys
import numpy as np

if len(sys.argv) != 2:
		print("Usage: python logreg_train.py dataset_train.csv")
		sys.exit(1)

dataset_file = sys.argv[1]
skip_cols = ["Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"]

X = []
y = []

with open(dataset_file, 'r') as f:
	reader = csv.DictReader(f)
	feature_names = [col for col in reader.fieldnames if col not in skip_cols]

	for row in reader:
		try:
			features = [float(row[f]) for f in feature_names]  # all must convert
			X.append(features)
			y.append(row["Hogwarts House"])
		except (ValueError, TypeError):
			# Skip this row entirely if any feature is non-numeric
			continue

# ---------> Using SGD
# model = LogisticRegression(learning_rate=0.1, epochs=1000, batch_size=1)

# ---------> Using Full Batch GD Aka normal GD
# model = LogisticRegression(learning_rate=0.1, epochs=1000)

# ---------> Using Mini-batch GD
model = LogisticRegression(learning_rate=0.1, epochs=1000, batch_size=32)

model.fit(X, y)
model.save_weights("weights.csv")
print("Training complete! Weights saved to weights.csv")