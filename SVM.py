import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('tkagg')
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    return data[:, :2], data[:, -1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

class SVM(object):
    def __init__(self, kernel, X_train, y_train, C=1, p=2, sigma=1):
        self.kernel = kernel
        self.X_train = X_train
        self.y_train = y_train
        self.m = X_train.shape[0]
        self.n = X_train.shape[1]
        self.alpha = np.random.random(self.m)
        self.K = self.ComputeKernel(kernel, X_train, p, sigma)
        self.C = C
        self.b = 0
        self.p = p
        self.sigma = sigma
        self.E = [ self._E(i) for i in range(self.m)]

    def ComputeKernel(self, kernel, X_train, p, sigma):
        if kernel == "linear":
            return np.dot(X_train, X_train.transpose())
        elif kernel == "poly":
            return (np.dot(X_train, X_train.transpose())+1) ** p
        elif kernel == "gaussian":
            G = np.dot(X_train, X_train.transpose())
            H = np.tile(np.diag(G), (self.m,1))
            return np.exp(-(H + H.T - 2*G)/(2*sigma**2))
        else:
            raise ValueError("kernel must be in linear, poly or gaussian")
    
    def _E(self, i):
        return self._g(i) - self.y_train[i]

    def _g(self, i):
        return np.dot(self.alpha * self.y_train, self.K[i][:]) + self.b
    
    def _KKT(self, i):
        y_g = self._g(i) * self.y_train[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    def select(self):
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        non_boundary = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_boundary)
        for i in index_list:
            if self._KKT(i):
                continue
            E1 = self.E[i]
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j 

    def _clip(self, val, L, H):
        if val<L:
            return L
        elif val>H:
            return H
        return val

    def train(self, max_iter=1):
        for _ in range(max_iter):
            i1, i2 = self.select()
            if self.y_train[i1] == self.y_train[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            
            eta = self.K[i1][i1] + self.K[i2][i2] - 2 * self.K[i1][i2]            
            if eta <= 0:
                continue
            alpha2_new_unc = self.alpha[i2] + self.y_train[i2] * (E1 - E2) / eta 
            alpha2_new = self._clip(alpha2_new_unc, L, H)
            alpha1_new = self.alpha[i1] + self.y_train[i1] * self.y_train[i2] * (self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.y_train[i1] * self.K[i1][i2] * (
                alpha1_new - self.alpha[i1]) - self.y_train[i2] * self.K[i2][i1] * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.y_train[i1] * self.K[i1][i2] * (
                alpha1_new - self.alpha[i1]) - self.y_train[i2] * self.K[i2][i1] * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)
    
    def predict(self, x):
        if self.kernel == "linear":
            r = np.dot((self.alpha*self.y_train).transpose(),np.dot(self.X_train,x.transpose())) + self.b
        elif self.kernel == "poly":
            r = np.dot((self.alpha*self.y_train).transpose(),(np.dot(self.X_train,x.transpose())+1)**self.p) + self.b
        elif self.kernel == "gaussian":
            d = 0
            for i in range(self.n):
                d += (self.X_train - x)[:,i] ** 2 
            d = d/(2*self.sigma**2)
            d = np.exp(-d)
            r = np.dot((self.alpha*self.y_train).transpose(),d) + self.b
        return 1 if r>0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)


res = []
for i in range(10):
    svm = SVM("gaussian", X_train, y_train)
    svm.train(300)
    res.append(svm.score(X_test, y_test))
print(np.mean(res))


