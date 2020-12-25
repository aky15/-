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

class AdaBoost(object):
    def __init__(self, X_train, y_train, n_estimators=3):
        self.X_train = X_train
        self.y_train = y_train
        self.n = X_train.shape[0]
        self.feature_dim = X_train.shape[1]
        self.W = np.ones((1,self.n))/self.n
        self.alpha = []
        self.classifiers = []
        self.n_estimators = n_estimators
    
    def train_classifier(self):
        min_error = np.inf
        split_feature = 0
        split_thrs = 0
        direction = 0
        for f_dim in range(self.feature_dim):
            for i in range(self.n):
                thrs = self.X_train[i][f_dim]

                predict = [ 1 if self.X_train[j][f_dim]<=thrs else -1 for j in range(self.n) ]
                error = np.sum(self.W * (predict != self.y_train))
                if error<min_error:
                    min_error = error
                    split_feature = f_dim
                    split_thrs = thrs
                    direction = 0
                    final_predict = predict

                predict = [ 1 if self.X_train[j][f_dim]>thrs else -1 for j in range(self.n) ]
                error = np.sum(self.W * (predict != self.y_train))
                if error<min_error:
                    min_error = error
                    split_feature = f_dim
                    split_thrs = thrs
                    direction = 1
                    final_predict = predict

        return min_error, split_feature, split_thrs, direction, final_predict

    def compute_alpha(self, error):
        return 1/2*np.log((1-error)/error)

    def train(self):
        for _ in range(self.n_estimators):
            min_error, split_feature, split_thrs, direction, predict = self.train_classifier()
            alpha = self.compute_alpha(min_error)
            self.alpha.append(alpha)
            self.W = self.W * np.exp(-alpha*self.y_train*predict)
            self.W /= np.sum(self.W)
            self.classifiers.append([split_feature, split_thrs, direction])

    def predict(self,x):
        res = 0
        for alpha, classifier in zip(self.alpha, self.classifiers):
            split_feature, split_thrs, direction = classifier
            if direction==0:
                if x[split_feature]<=split_thrs:
                    res += alpha
                else:
                    res -= alpha
            elif direction==1:
                if x[split_feature]>split_thrs:
                    res += alpha
                else:
                    res -= alpha
        return 1 if res>0 else -1

adaboost = AdaBoost(X_train, y_train, 12)
adaboost.train()
prediction= []
for x in X_test:
    prediction.append(adaboost.predict(x))
print(np.sum(prediction==y_test)/y_test.shape[0])
