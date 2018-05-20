import numpy as np

class Perceptron(object):
    """
    Perceptron Classifier
    @param eta: float = Learning rate (between 0.0 and 1.0)
    @param n_iter: Int = passes over the training set
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """
        Fit training data
        @param X: Array-like, shape = [number of samples, number of features]
        @param Y: Array-like, shape = [number of samples]
        @return self: object
        """
        self.w_ = np.zeros(1 + X.shape[1]) # create numpy array of zeros of length = (1 + number of features) to be used for weights
        self.errors_ = []
        for _ in range (self.n_iter):
            errors = 0 
            for xi, yi in zip (X,y):    # Iterating over each pair of xi and yi
                update = self.eta * (yi - self.predict(xi))
                self.w_[1:] += update * xi  # Updating feature weights
                self.w_[0] += update # updating intercept weight
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """
        Calculate net input
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """
        @param X: Array-like, shape  = [number of samples, number of features]
        @return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

################################################################################################################################################

class AdalineGD(object):
    """
    ADAptive LInear NEuron classifier
    @param eta: float - Learning rate (between 0.0 and 1.0)
    @param n_iter: int - Passes over the training sets
    """
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """
        Fit training data
        @param X: Array-like, shape = [number of samples, number of features]
        @param Y: Array-like, shape = [number of samples]
        @return self: object
        """
        self.w_ = np.zeros(1 + X.shape[1]) # Initializing weight vector of length = (1 + number of features)
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
            
    def net_input(self,X):
        """
        Calculate net input
        """
        return np.dot(X, self.w_[1:] + self.w_[0])
    
    def activation(self, X):
        """
        Compute linear activation
        """
        return self.net_input(X)
    
    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.activation(X) > 0.0, 1, -1)
