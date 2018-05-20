import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.linear import AdalineGD
print('Loading input file')
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.head()
y = df.iloc[0:100,4]
y = np.where(y=='Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values
print("Standardizing datasets")
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std() # standardization = X(j)new = (X(j) - mean(j)) / std_dev(j)
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std() # X(j) = vector consisting of the jth feature values of all training samples n
print("Training model")
ada = AdalineGD(eta = 0.01, n_iter = 10)
ada.fit(X_std,y)
