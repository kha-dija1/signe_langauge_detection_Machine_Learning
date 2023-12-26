import time
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from signe_langauge.algos.preproces import get_data

X, Y = get_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

algos = ['KNN', 'Perceptron', 'SVM', 'TREE', 'RF', 'LR']
colors = ['#FFC107', '#E91E63', '#9C27B0', '#3F51B5', '#009688', '#FF5722']


def evaluate_algorithm(clf, algo_name):
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()

    execution_time = end_time - start_time
    accuracy = clf.score(x_test, y_test) * 100
    print(f"Accuracy for {algo_name}: {accuracy:.2f}")
    print(f"Execution time for {algo_name}: {execution_time:.2f} seconds")

    return accuracy, execution_time


classifiers = [
    KNeighborsClassifier(),
    Perceptron(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression()
]

accuracy_scores = []
execution_times = []

for algo, clf in zip(algos, classifiers):
    accuracy, execution_time = evaluate_algorithm(clf, algo)
    accuracy_scores.append(accuracy)
    execution_times.append(execution_time)

# Plot the accuracy scores
plt.bar(algos, accuracy_scores, color=colors)
plt.title('Accuracy: Algorithms Features extraction /LBP')
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
plt.title('Execution Time Features extraction /LBP')
plt.xlabel('Algorithms')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45)

# Add execution times on top of each bar
for i, time_val in enumerate(execution_times):
    plt.text(i, time_val, f'{time_val:.2f}s', ha='center', va='bottom')

plt.show()
