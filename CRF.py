import numpy as np
T2 = np.array([[0.6, 1],[1,0]])
T3 = np.array([[0, 1],[1,0.2]])
S1 = np.array([[1],[0.5]])
S2 = np.array([[0.8,0.5]])
S3 = np.array([[0.8,0.5]])

y = [1,2,2]
logit = T2[y[0]-1][y[1]-1] + T3[y[1]-1][y[2]-1] + S1[0][y[0]-1] + S2[0][y[1]-1] + S3[0][y[2]-1]

M1 = S1
M2 = T2 + S2
M3 = T3 + S3
print(M1)
print(M2)
print(M3)
print(np.exp(logit))
print(np.exp(3.2))
print(np.exp(M1[0][y[0]-1] +M2[y[0]-1][y[1]-1] + M3[y[1]-1][y[2]-1] ))