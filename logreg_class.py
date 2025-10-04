import csv
from colorama import init, Fore, Style
import numpy as np

init(autoreset=True)

class LogisticRegression():
	def __init__(self, learning_rate=0.01, epochs=1000, batch_size=None): # epochs mean the number of iterations, batch_size means the number of samples in each batch None means normal gradient descent:
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.weights = None
		self.n_classes = 4

	@staticmethod
	def _sigmoid(z):
		return 1 / (1 + np.exp(-z))

	def preprocess_data(self, X):
		# Add bias term (column of ones) to the input features
		X = np.array(X, dtype=float)
		X_min = X.min(axis=0)
		X_max = X.max(axis=0)
		X = (X - X_min) / (X_max - X_min) # Normalize features
		bias = np.ones((X.shape[0], 1))
		X = np.hstack((bias, X))
		return X
	def encoded_labels(self, y):
		# One-hot encode the labels
		mapping = {'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}
		return np.array([mapping[label] for label in y])
	
	def fit(self, X, y):
		X = self.preprocess_data(X)
		y_encoded = self.encoded_labels(y)

		n_samples, n_features = X.shape
		self.weights = np.zeros((self.n_classes, n_features))

		if self.batch_size is None:
			opt_type = Fore.CYAN + "Full-Batch Gradient Descent (BGD)" + Style.RESET_ALL
		elif self.batch_size == 1:
			opt_type = Fore.MAGENTA + "Stochastic Gradient Descent (SGD)" + Style.RESET_ALL
		else:
			opt_type = Fore.YELLOW + f"Mini-Batch Gradient Descent (MBGD, batch_size={self.batch_size})" + Style.RESET_ALL

		print(f"Training used {opt_type} for {self.epochs} epochs with learning rate {self.learning_rate}")

		for class_idx in range(self.n_classes):
			y_binary = (y_encoded == class_idx).astype(int)
			w = np.zeros(n_features)
		
			for epoch in range(self.epochs):
				if self.batch_size is None:
					# Full batch gradient descent
					predictions = self._sigmoid(np.dot(X, w))
					errors = predictions - y_binary
					gradient = np.dot(X.T, errors) / n_samples
					w -= self.learning_rate * gradient

				else:
					# Mini-batch gradient descent
					for start in range(0, n_samples, self.batch_size):
						end = start + self.batch_size
						X_batch = X[start:end]
						y_batch = y_binary[start:end]
						predictions = self._sigmoid(np.dot(X_batch, w))
						errors = predictions - y_batch
						gradient = np.dot(X_batch.T, errors) / len(y_batch)
						w -= self.learning_rate * gradient

			self.weights[class_idx] = w
		print(Fore.GREEN + "\nTraining complete! 🎉 Weights @ weights.csv" + Style.RESET_ALL)

	def save_weights(self, filename):
		with open(filename, 'w', newline='') as f:
			writer = csv.writer(f)
			for row in self.weights:
				writer.writerow(row)
