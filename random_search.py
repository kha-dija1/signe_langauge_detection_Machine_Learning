import time
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from signe_langauge.algos.preproces import get_data

X, Y = get_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

algos = ['KNN', 'Perceptron', 'SVM', 'TREE', 'RF', 'LR']
colors = ['#FFC107', '#E91E63', '#9C27B0', '#3F51B5', '#009688', '#FF5722']

def random_search_all_algorithms():
    accuracy = []  # List to store the accuracy scores
    execution_times = []  # List to store the execution times

    # Define the parameter distributions for each algorithm
    param_dist = [
        {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
        {'alpha': [0.0001, 0.001, 0.01], 'penalty': ['l1', 'l2', 'elasticnet']},
        {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'degree': [2, 3, 4]},
        {'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10]},
        {'n_estimators': [100, 200, 300], 'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10]},
        {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
    ]

    # Define the classifiers for each algorithm
    classifiers = [
        KNeighborsClassifier(),
        Perceptron(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        LogisticRegression()
    ]

    # Perform random search for each algorithm and store the accuracy scores and execution times
    for algo, clf, param_dist in zip(algos, classifiers, param_dist):
        start_time = time.time()
        random_search = RandomizedSearchCV(clf, param_dist, n_iter=10, cv=5, scoring='accuracy')
        random_search.fit(x_train, y_train)
        end_time = time.time()

        execution_time = end_time - start_time
        execution_times.append(execution_time)
        accuracy.append(random_search.best_score_ * 100)
        print(f"Best parameters for {algo}: {random_search.best_params_}")
        print(f"Best accuracy for {algo}: {random_search.best_score_}")
        print(f"Execution time for {algo}: {execution_time:.2f} seconds")



    return accuracy, execution_times

# Call the function to perform random search on all algorithms
accuracy_scores, execution_times = random_search_all_algorithms()

# Plot the accuracy scores
plt.bar(algos, accuracy_scores, color=colors)
plt.title('Accuracy: Algorithms Random Search  /LBP')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy %')
plt.ylim(0, 100)
plt.xticks(rotation=45)

# Add accuracy scores on top of each bar
for i, score in enumerate(accuracy_scores):
    plt.text(i, score, f'{score:.2f}', ha='center', va='bottom')

plt.show()

# Plot the execution times
plt.bar(algos, execution_times, color=colors)
plt.title('Execution Time for Random Search  /LBP')
plt.xlabel('Algorithms')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45)

# Add execution times on top of each bar
for i, time_val in enumerate(execution_times):
    plt.text(i, time_val, f'{time_val:.2f}s', ha='center', va='bottom')

plt.show()

