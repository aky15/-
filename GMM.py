import numpy as np
class GMM(object):
    def __init__(self, n_components, X_train):
        self.m = len(X_train)
        self.n_components = n_components
        self.gamma = np.ones((self.m, n_components))/(self.m*n_components)
        self.mu = np.random.random((n_components, 1))*np.mean(X_train)
        self.sigma2 = np.ones((n_components, 1))*(np.mean(np.dot(X_train.transpose(),X_train))/self.m)
        self.alpha = np.ones((n_components, 1))/n_components
        self.X_train = X_train
    
    def phi(self, x, mu, sigma2):
        return 1/(np.sqrt(2*np.pi*sigma2)) * np.exp(-(x-mu)**2/(2*sigma2))
    
    def E(self):
        for k in range(self.n_components):
            self.gamma[:,k] = self.alpha[k][0] * self.phi(self.X_train, self.mu[k][0], self.sigma2[k][0]).ravel()
        self.gamma /= np.sum(self.gamma, axis=1).reshape(-1,1)
    
    def M(self):
        self.mu = (np.dot(self.gamma.transpose(), self.X_train).ravel() / np.sum(self.gamma, axis = 0)).reshape((-1,1))
        for k in range(self.n_components):
            self.sigma2[k] = np.dot(self.gamma[:,k].reshape(1,-1), (self.X_train - self.mu[k][0])**2).ravel() / np.sum(self.gamma[:,k])
        self.alpha = (np.sum(self.gamma, axis=0)/self.m).reshape(-1,1)

    def train(self):
        while 1:
            last = self.mu
            self.E()
            self.M()
            if np.sum((self.mu - last)**2)<0.1:
                break

data = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]).reshape(-1, 1)
gmm = GMM(2,data)
gmm.train()
print(gmm.mu)
print(gmm.sigma2)