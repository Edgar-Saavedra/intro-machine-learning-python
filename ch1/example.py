import numpy as np

# core of numpy is ndarray --
# all elements of  the array mus be of the same type!
x = np.array([
  [1, 2, 3],
  [4, 5, 6]
])
print(f"x:\n{x}")

# SciPy --
# collection of fucntions for scientific computing
# scikit-learn draws from scipy collections of functions for implementing
# algorithms. Most important part of SciPy is scipy.sparse, provides sparse matrices
# spase matrices used whenever we want to stor a 2D array with mostly zeros
from scipy import sparse

# a 2d NumPy array with a diagonal of ones and zeros everywhere else
eye = np.eye(4)
print(f"Numpy array:\n", eye)
# conver numpy array to scipy sparse matrix in CSR format
# only nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix: \n", sparse_matrix)

# way to create the same sparse matrix as before, using OOO format
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n", eye_coo)

# matplotlib
# primary scientific plotting library
# https://matplotlib.org/3.2.2/users/shell.html
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker = "x")
plt.show()
# Pandas
# built around the DataFrame that is modeled after the R DataFrame
# allows SQL like queries and joins of tables
import pandas as pd
# create a simple dataset of people
data = {
  'Name': ["John", "Anna", "Peter", "Linda"],
  'Location': ["New York", "Paris", "Berlin", "London"],
  'Age': [24, 13, 53, 33]
}
data_pandas = pd.DataFrame(data)

import IPython as ip
ip.display.display(data_pandas)
#  select all rows that have an age column greater than 30
ip.display.display(data_pandas[data_pandas.Age > 30])

# mglearn
import mglearn