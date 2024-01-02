# Name : Onkar Eknath Shelar
# Email ID - os9660@rit.edu

import pandas as pd
import numpy as np
import math

class my_Logistic:
    def __init__(self, learning_rate=0.1, batch_size=10, max_iter=100, shuffle=False):
        # Initializing hyperparameters for logistic regression
        self.learning_rate = learning_rate  # Setting the learning rate for updating weights
        self.batch_size = batch_size  # Specifying the number of data points in each batch
        self.max_iter = max_iter  # Defining the maximum number of iterations (epochs)
        self.shuffle = shuffle  # Indicating whether the data is being shuffled in each epoch

    def fit(self, X, y):
        data = X.to_numpy()  # Converting the input data to a numpy array
        data = np.concatenate([data, np.ones(shape=(data.shape[0], 1))], axis=1)  # Adding a column of ones for bias
        d = data.shape[1]  # Calculating the number of features (including bias)
        self.w = np.zeros(d)  # Initializing weights as zeros
        for i in range(self.max_iter):  # Iterating through epochs
            if self.shuffle:
                inds = np.random.permutation(range(len(data)))  # Shuffling the data indices
            else:
                inds = np.arange(len(data))  # Using a range of indices if not shuffling

            num_inds = len(inds)

            while num_inds > 0:
                num_inds -= self.batch_size

                if num_inds > self.batch_size:
                    b_inds = inds[:self.batch_size]  # Taking a batch of indices
                    inds = inds[self.batch_size:]  # Removing the batch indices from the list
                else:
                    b_inds = inds  # Using the remaining indices if less than a full batch

                Xb, yb = data[b_inds], y[b_inds]  # Getting the batch of data
                self.batch_update(Xb, yb)  # Updating weights based on the batch

    def batch_update(self, Xb, yb):
        gradient = self.compute_gradient(wts=self.w, Xb=Xb, yb=yb)  # Computing the gradient of the loss
        self.w -= self.learning_rate * gradient  # Updating weights using the learning rate

    def predict_proba(self, X):
        X = np.concatenate([X, np.ones(shape=(X.shape[0], 1))], axis=1)  # Adding a column of ones for bias
        probs = self.sigmoid(np.dot(X, self.w))  # Calculating prediction probabilities
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)  # Getting prediction probabilities
        predictions = [1 if prob >= 0.5 else 0 for prob in probs]  # Making binary predictions
        return predictions

    def compute_gradient(self, wts, Xb, yb):
        m = len(yb)
        predicted = self.sigmoid(np.dot(Xb, wts))
        error = predicted - yb
        gradient = 2 * (1 / m) * np.dot(Xb.T, error * predicted * (1 - predicted))
        return gradient
    # Compute the gradient of the loss

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # Calculating the sigmoid function for logistic regression

