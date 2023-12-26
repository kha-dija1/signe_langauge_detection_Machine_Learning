from matplotlib import image as mpimg, pyplot as plt
import SVM
import knnAlgo
import perceptron
import preproces
import TREE
import  regression_logistic
import Random_forest

from skimage.feature  import hog
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
accuracy =[]
from sklearn.model_selection import cross_val_score
# Split data into training and testing sets
X,Y=preproces.get_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
algos= ['K Nearest Neighbors', 'Perceptron', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
           'Logistic regression']
colors = ['#FFC107', '#E91E63', '#9C27B0', '#3F51B5', '#009688', '#FF5722']

ac=[knnAlgo.knn_accuracy,perceptron.accuracy_score_p,SVM.svm_accuracy,TREE.tree_accuracy,
    Random_forest.random_forest_accuracy,regression_logistic.accuracy_RL]

for i in range(len(ac)):
    accuracy.append(ac[i]()*100)
    print(accuracy)
# Plotting the accuracy values

algos= ['KNN', 'Perceptron', 'SVM', 'TREE', 'RF',
           'LR']
colors = ['#FFC107', '#E91E63', '#9C27B0', '#3F51B5', '#009688', '#FF5722']

# Customizing the plot
plt.bar(algos, accuracy, color=colors)
plt.title('Accuracy Algorithms KNN OPtimisation /Elbow_methode  Feature extraction / HOG ')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy %')
plt.ylim(0, 100)  # Setting the y-axis limits
plt.xticks(rotation=45)  # Rotating the x-axis labels for better readability

plt.show()
# dont use cv2.imread  mpimg.imread is bett
#img = preproces('C:/Users/hp/Downloads/q.png')

#print(tree_accuracy())
#print("TREE THINK THIS IMAGE IS ",TREE(img))
#print("perceptron THINK THIS IMAHGE IS" ,perceptron(img))
#print("KNN THINK THIS IMAH+GE IS" ,KNN(img))
#print("SVM THINK THIS IMAGE IS ",SVM(img)[0])

#print("RL THINK THIS IMAHGE IS" ,RL(img))
#print("RF think this is ",RF(img))

