from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

from signe_langauge.algos.preproces import get_data
X,Y=get_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


def svm_accuracy(param= {'C':1, 'kernel': 'linear', 'degree': 2}):
    # Create an SVM classifier
    clf= SVC(kernel=param['kernel'], C=param['C'], random_state=param['degree'])
    # Train the classifier on the training data
    clf.fit(x_train, y_train)

    # Predict the labels of the test data
    y_pred = clf.predict(x_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
def svm(inputx,param= {'C':1, 'kernel': 'linear', 'degree': 2}):# Entraîner le modèle SVM
 svm = SVC(kernel=param['kernel'], C=param['C'], random_state=param['degree'])
 svm.fit(x_train, y_train)

 # Prédire les classes de l'ensemble de test
 outputy = svm.predict([inputx])
 return outputy[0]

def grid_search_on_SVM():
    # Define the hyperparameters and their respective values to test
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'degree': [2, 3, 4]}

    # Create a GridSearchCV object with the SVM model, hyperparameters, and cross-validation settings
    svm_model = SVC()
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')

    # Fit the GridSearchCV object to the training data to find the best hyperparameters
    grid_search.fit(x_train, y_train)

    # Print the best hyperparameters and their respective performance metric
    print("Best hyperparameters for grid_search_on_SVM: ", grid_search.best_params_)
    print("Best accuracy grid_search_on_SVM: ", grid_search.best_score_)
    accuracies = grid_search.cv_results_['mean_test_score'] * 100
    plt.plot(range(1, len(accuracies) + 1), accuracies, color='#E91E63')
    plt.title("Accuracy: Algorithms SVM Optimization (Grid Search)")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy %")

    plt.show()
    return grid_search.best_params_

def SVM(inputx):
    #x=grid_search_on_SVM()
    return svm(inputx)
