from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from signe_langauge.algos.preproces import get_data

X, Y = get_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


def random_forest_accuracy(param={'n_estimators': 100, 'max_depth': None}):
    # Create a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=param['n_estimators'], max_depth=param['max_depth'], random_state=42)

    # Train the classifier on the training data
    clf.fit(x_train, y_train)

    # Predict the labels of the test data
    y_pred = clf.predict(x_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def random_forest(inputx, param={'n_estimators': 100, 'max_depth': None}):
    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=param['n_estimators'], max_depth=param['max_depth'], random_state=42)
    model.fit(x_train, y_train)

    # Predict the classes of the input data
    outputy = model.predict([inputx])

    return outputy[0]


def grid_search_on_random_forest():
    # Define the hyperparameters and their respective values to test
    param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}

    # Create a GridSearchCV object with the Random Forest model, hyperparameters, and cross-validation settings
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')

    # Fit the GridSearchCV object to the training data to find the best hyperparameters
    grid_search.fit(x_train, y_train)

    # Print the best hyperparameters and their respective performance metric
    print("Best hyperparameters for grid_search_on_random_forest: ", grid_search.best_params_)
    print("Best accuracy grid_search_on_random_forest: ", grid_search.best_score_)
    accuracies = grid_search.cv_results_['mean_test_score'] * 100
    plt.plot(range(1, len(accuracies) + 1), accuracies, color="#009688")
    plt.title("Accuracy: Algorithms Random Forest Optimization (Grid Search)")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy %")

    plt.show()
    return grid_search.best_params_


def RF(inputx):
    #best_params = grid_search_on_random_forest()
    return random_forest(inputx)
