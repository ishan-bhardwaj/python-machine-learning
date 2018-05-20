import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from models.linear import Perceptron

print('Laoding input file')
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

df.head()

y = df.iloc[0:100,4]

y = np.where(y=='Iris-setosa', -1, 1)

X = df.iloc[0:100, [0,2]].values

print('Plotting values')

plt.scatter(X[:50,0], X[:50,1], color = 'red', marker='o', label='setosa')

plt.scatter(X[50:100,0], X[50:100,1], color = 'blue', marker='x', label='versicolor')

plt.xlabel('petal length')

plt.ylabel('sepal length')

plt.legend(loc='upper left')

plt.show()

print('Training model')

ppn = Perceptron(eta = 0.1, n_iter = 10)

ppn.fit(X,y)

print('Plotting missclassifications')

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')

plt.xlabel('Epochs')

plt.ylabel('Number of missclassifications')

plt.show()
