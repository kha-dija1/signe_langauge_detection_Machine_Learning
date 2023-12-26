from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from signe_langauge.algos.preproces import get_data
X,Y=get_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def train_perceptron(X, y):

    perceptron = Perceptron()
    perceptron.fit(x_train, y_train)
    return perceptron, x_test, y_test

def predict_labels(perceptron, X_test):
    return perceptron.predict(X_test)
def accuracy_score_p():
    # Train the perceptron model
    perceptron = Perceptron()
    perceptron.fit(x_train, y_train)

    # Predict the labels for the test set
    y_pred = perceptron.predict(x_test)

    # Calculate the accuracy score
    score = accuracy_score(y_test, y_pred)

    return score
def calculate_accuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred)

def calculate_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)

def perceptron(image):
    perceptron = Perceptron()
    perceptron.fit(x_train, y_train)

    predicted_class = perceptron.predict([image])[0]
    return predicted_class

