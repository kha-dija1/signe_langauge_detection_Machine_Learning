from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from signe_langauge.algos.preproces import get_data
X,Y=get_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
def tree_accuracy():
    # Create the classifier with default hyperparameters
    dtc = DecisionTreeClassifier()

    # Train the classifier on the training data
    dtc.fit(x_train, y_train)

    # Predict the labels for the test data
    y_pred = dtc.predict(x_test)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return  accuracy
def tree(inputx,param= {'max_depth':10, 'min_samples_split': 3, 'min_samples_leaf': 1}):
    # Create the classifier with default hyperparameters
    dtc = DecisionTreeClassifier(max_depth=param['max_depth'], min_samples_split=param['min_samples_split'],
                            min_samples_leaf=param['min_samples_leaf'])

    # Train the classifier on the training data
    dtc.fit(x_train, y_train)

    # Predict the labels for the test data
    y_pred = dtc.predict([inputx])

    return y_pred[0]
def grid_search_on_tree():
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 4, 6, 8, 10],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10]
    }
    # Create a decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)

    # Create a GridSearchCV object to find the best hyperparameters
    grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)

    # Train the model with the grid search object
    grid_search.fit(x_train, y_train)

    # Print the best hyperparameters and the accuracy score on the test set
    print("Best hyperparameters:", grid_search.best_params_)
    print("Accuracy on test set:", grid_search.score(x_test, y_test))
    return grid_search.best_params_

def TREE(inputx):
    #x=grid_search_on_tree()
    return tree(inputx)

