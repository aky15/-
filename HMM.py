import numpy as np
import torch

class HMM(object):
    def __init__(self):
        self.alpha = None
        self.beta = None

    def forward(self, Q, V, A, B, PI, O):
        T = len(O)
        N = len(Q)
        alpha = np.zeros((T,N))
        for t in range(T):
            IndexOfO = V.index(O[t])
            if t==0:
                alpha[t] = PI * B[:,IndexOfO]
            else:
                for i in range(N):
                    alpha[t][i] = np.dot(alpha[t-1][:],A[:,i])*B[i,IndexOfO]
        self.alpha = alpha
        return np.sum(alpha[-1][:])
    
    def backward(self, Q, V, A, B, PI, O):
        T = len(O)
        N = len(Q)
        beta = np.ones((T,N))        
        for t in range(T-2,-1,-1):
            for i in range(N):
                IndexOfOP1 = V.index(O[t+1])
                beta[t][i] = np.sum(A[i,:] * B[:,IndexOfOP1] * beta[t+1,:])
        IndexOfOP1 = V.index(O[0])
        self.beta = beta
        return np.sum(PI * B[:,IndexOfOP1] * beta[0,:])
    
    def viterbi(self, Q, V, A, B, O, PI):
        T = len(O)
        N = len(Q)  
        delta = np.ones((T,N))     
        phi = np.zeros((T,N))
        IndexOfO = V.index(O[0])
        delta[0,:] = PI * B[:,IndexOfO]
        for t in range(1,T):
            IndexOfO = V.index(O[t])
            for i in range(N):
                da = delta[t-1,:] * A[:,i]
                delta[t,i] = np.max(da)*B[i,IndexOfO]
                phi[t,i] = da.tolist().index(np.max(da))
        path = [0] * T
        for t in range(T-1,-1,-1):
            if t==T-1:
                tmp = delta[t,:].tolist()
                path[t] = int(tmp.index(np.max(delta[t,:])))+1
            else:
                path[t] = int(phi[t+1,path[t+1]-1])+1
        return path



Q = np.array([1, 2, 3])
V = ['红', '白']
A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
O = np.array(['红', '白', '红','白'])
PI = np.array([0.2, 0.4, 0.4])

hmm = HMM()
p = hmm.forward(Q, V, A, B, PI, O)
p = hmm.backward(Q, V, A, B, PI, O)
print(hmm.viterbi(Q, V, A, B, O, PI))