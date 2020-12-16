import numpy as np
import collections

class MaxEntropy(object):
    def __init__(self, max_iter=200):
        self.max_iter = max_iter
        self.Ph_X = collections.defaultdict(float)
        self.Ph_XY = collections.defaultdict(float)
        self.fij = collections.defaultdict(int)
        self.EP_h = collections.defaultdict(float)
        self.w = collections.defaultdict(float)
        self.Y = set()
        self.M = 0

    def Py_x(self, target_j, x):
        num = 0
        den = 0
        for i,j in self.fij:
            if i in x and j==target_j:
                num += self.w[(i,j)]
        num = np.exp(num)
        den = collections.defaultdict(float)
        for i,j in self.fij:
            if i in x:
                den[j] += self.w[(i,j)]
        den_sum = 0
        for j in den:
            den_sum += np.exp(den[j])
        return num/den_sum

    def LoadData(self, samples):
        for sample in samples:
            x = sample[1:]
            y = sample[0]
            self.Ph_X[tuple(x)] += 1
            self.Ph_XY[(tuple(x),y)] += 1  
            self.Y.add(y)        
            for i in x:
                if not (i,y) in self.fij:
                    self.fij[(i,y)] = 1
                    self.w[(i,y)] = np.random.random()
            self.M = len(self.fij)

        for x in self.Ph_X:
            self.Ph_X[x] /= len(samples)
        for xy in self.Ph_XY:
            self.Ph_XY[xy] /= len(samples)
        for i,j in self.fij:
            for x,y in self.Ph_XY:
                if i in x and j==y:
                    self.EP_h[(i,j)] += self.Ph_XY[(x,y)]

    def train(self):
        iter = 0
        last = self.w.copy()
        while iter<self.max_iter:
            for i,j in self.fij:
                EP = 0
                for x in self.Ph_X:
                    if i in x:
                        EP += self.Ph_X[x] * self.Py_x(j,x)
                self.w[(i,j)] += 1/self.M * np.log(self.EP_h[(i,j)]/EP)
            #print(np.sum(np.abs(np.array(list(self.w.values()))-np.array(list(last.values())))))
            last = self.w.copy()
            iter+=1
    
    def predict(self, x):
        result = collections.defaultdict(float)
        for j in self.Y:
            result[j] = self.Py_x(j,x)
        return max(result, key = result.get)

dataset = np.array([['no', 'sunny', 'hot', 'high', 'FALSE'],
           ['no', 'sunny', 'hot', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'high', 'FALSE'],
           ['yes', 'rainy', 'mild', 'high', 'FALSE'],
           ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
           ['no', 'rainy', 'cool', 'normal', 'TRUE'],
           ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
           ['no', 'sunny', 'mild', 'high', 'FALSE'],
           ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
           ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
           ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
           ['yes', 'overcast', 'mild', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
           ['no', 'rainy', 'mild', 'high', 'TRUE']])

maxent = MaxEntropy()
maxent.LoadData(dataset)
maxent.train()
x = ['overcast', 'mild', 'high', 'FALSE']
#print(maxent.predict(x))
for x in dataset:
    print(maxent.predict(x[1:])==x[0])