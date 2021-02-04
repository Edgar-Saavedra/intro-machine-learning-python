# these imports are assumed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
# END: these imports are assumed

from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("Keys of iris_dataset: \n", iris_dataset.keys())
print(iris_dataset['DESCR'][:193]+"\n")
print("Target names: ", iris_dataset['target_names'])
# Target names:  ['setosa' 'versicolor' 'virginica']
print("Feature names:\n", iris_dataset['feature_names'])
print("Type of data:", type(iris_dataset['data']))
print("Shape of data:", iris_dataset['data'].shape)
# contains measurement for 150 flowers.
# Items are called samples
# properties Features
# the shape is the number of samples times the number of features

print("First five rows of data:\n", iris_dataset['data'][:5])
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# First five rows of data:
#  [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]]

print("Type of target", type(iris_dataset['target']))
print("Shape of target", iris_dataset['target'].shape)
print("Target:\n", iris_dataset['target'])
# Target:
#  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
# ['setosa' 'versicolor' 'virginica']
# 0 = setosa
# 1 = versicolor
# 2 = virginica

# will model "generalize"
# show it new data to asses performance
# training data - data used to build our machine learning model
# training set - test data, test set, hod-out set
# scikit-learn has function that shuffles dataset and splits it: train_test_split()
# scikit-learn data denoted capital X (X cuz data is 2-D)
# scikit-learn labels denoted lowercase y (lowercase 1-D)

from sklearn.model_selection import train_test_split
# random_state makes outcome deterministic, will always have same outcome
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# IMPORTANT STEP: First make sure to look at data:
# Visualize it using scatter plot
# convert Numpy array to pandas DataFrame
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scattermatrix form the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()

# many classification algorithms in scikit-learn
# ** k-nearest neighbors classifier **
# only consists of storing the training set
# finds the point in the training set that is closest to the new point
# assigsn the label of this training point to the new data point
# k in k-nearest instead of using the only the closes neighbor, 
# we can consider any fixed number k of neighbors
# we can make a prediction using the majority class amongst these neighbors

# all machine learning models in scikit-learn implement teir own classes
# called Estimator classes
# k-nearest uses KNeighborsClassifier class in neighbors module

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
# to build the model on the training set, we call the fit method of the
# knn object. X_train containing training data and y_train the training labels
# fit returns the knn object itself
knn.fit(X_train, y_train)

# IMPORTANT: Making predictions
# we can make predictions
# we found an iris in the whild with 
# sepal length - 5 cm
# sepal width - 2.9 cm
# petal length - 1 cm
# petal width - 0.2 cm

# scikit-learn alwys expects 2-D arrays for the data
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape", X_new.shape)

# to make a perdiction we call the perdict method
perdiction = knn.predict(X_new)
print("Perdiction:", perdiction)
print("Predicted target name:", iris_dataset['target_names'][perdiction])

# IMPORTANT: Computing accuracy
y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)
print(f"Test set score: {knn.score(X_test, y_test):.2f}")

