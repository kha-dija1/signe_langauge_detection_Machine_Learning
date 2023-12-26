from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from signe_langauge.algos.preproces import get_data
X,Y=get_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


def RL(x):
    # Perform grid search to find the best hyperparameters

    # Train the decision tree classifier with the best hyperparameters
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(x_train, y_train)

    # Predict the class of the input image
    y_pred = logreg.predict([x])
    return y_pred[0]
def accuracy_RL():
    # Train the logistic regression model
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(x_train, y_train)

    # Predict the labels for the test set
    y_pred = logreg.predict(x_test)

    # Calculate the accuracy score
    score = accuracy_score(y_test, y_pred)

    return score