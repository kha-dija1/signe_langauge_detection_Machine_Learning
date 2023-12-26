from sklearn.datasets import load_iris

from preproces import get_data
import statistics
from random import randint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier


X,Y=get_data()

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Load the iris dataset

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#print(x_train[0],y_train[0])
def knn_accuracy(k):

 # Create KNN classifier with k=k
 knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto', metric='euclidean')

 # Train the classifier on the training data
 knn.fit(x_train, y_train)

 # Use the trained classifier to predict the test data
 y_pred = knn.predict(x_test)

 # Calculate the accuracy of the classifier
 accuracy = accuracy_score(y_test, y_pred)

 print("Accuracy:", accuracy)
 return accuracy
def knn(k,inputx):

 # Create KNN classifier with k=k
 knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto', metric='euclidean')


 # Train the classifier on the training data
 knn.fit(x_train, y_train)

 # Use the trained classifier to predict the test data
 y_pred = knn.predict([inputx])

 # Calculate the accuracy of the classifier
 #accuracy = accuracy_score(y_test, y_pred)

 #print("Accuracy:", accuracy)
 return y_pred[0]


def elbow_methode():
    Scores = []
    for i in range(2, 10):
        Scores.append(knn_accuracy(i))
    #plt.plot(Scores)
    #plt.show()
    max_value = max(Scores)
    max_index = Scores.index(max_value)
    max_index += 2
    return max_index
def cross_validation():
 cv_scores = []
 # Define range of k values to test
 k_range = range(2, 10)
 # Perform cross-validation for each value of k
 for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  scores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
  cv_scores.append(scores.mean())

 # Plot the cross-validation scores for each k
 #plt.plot(k_range, cv_scores)
 #plt.xlabel('Number of Neighbors (k)')
 #plt.ylabel('Cross-Validation Accuracy')
 #plt.show()

 # Identify the optimal value of k (i.e. elbow point)
 optimal_k = k_range[cv_scores.index(max(cv_scores))]
 return optimal_k


def random_search():
    # Define parameter grid to search over
    param_dist = {"n_neighbors": randint(2, 10),
                  "weights": ["uniform", "distance"],
                  "metric": ["euclidean", "manhattan"]}

    # Create KNN classifier
    knn = KNeighborsClassifier()

    # Perform random search over parameter grid
    rand_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy')
    rand_search.fit(x_train, y_train)

    # Print results
    print("Best parameters: ", rand_search.best_params_)
    print("Best accuracy: ", rand_search.best_score_)

    return rand_search.best_params_['n_neighbors']


def find_best_k():
    the_k = []
    #the_k.append(cross_validation())
    the_k.append(elbow_methode())

    the_k.append(random_search())
    return statistics.mode(the_k)
def KNN(inputx):
    x=find_best_k()
    return knn(3,inputx)


