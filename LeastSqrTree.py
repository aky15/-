import numpy as np

class Node(object):
    def __init__(self, idx=None, split=None, val=None):
        self.idx = idx
        self.split = split
        self.val = val
        self.left = None
        self.right = None

def distance(idx, split, feat, label):   
    cate1 = label[feat[:,idx]<=feat[split][idx]]
    cate2 = label[feat[:,idx]>feat[split][idx]]
    distance = np.sum((cate1 - np.mean(cate1))**2) + np.sum((cate2 - np.mean(cate2))**2)
    return distance

def split(feat,label):
    MinLoss = np.inf
    for idx in range(len(feat[0])):
        for split in range(len(feat)):
            loss = distance(idx, split, feat, label)
            if loss < MinLoss:
                MinLoss = loss
                min_idx = idx
                min_split = split
    return min_idx,min_split,MinLoss

def build(feat, label, ep):
    if len(feat) == 0:
        return None
    min_idx,min_split,MinLoss = split(feat, label)
    root = Node(min_idx,feat[min_split][min_idx],np.mean(label))
    if MinLoss < ep:
        root.left = Node(None,None,np.mean(label[feat[:,min_idx]<=feat[min_split][min_idx]]))
        root.right = Node(None,None,np.mean(label[feat[:,min_idx]>feat[min_split][min_idx]]))
        return root
    root.left = build(feat[feat[:,min_idx]<=feat[min_split][min_idx]], label[feat[:,min_idx]<=feat[min_split][min_idx]], ep)
    root.right = build(feat[feat[:,min_idx]>feat[min_split][min_idx]], label[feat[:,min_idx]>feat[min_split][min_idx]], ep)
    return root

def dfs(node,feat):
    if node.idx==None:
        return node.val
    if feat[node.idx]<=node.split:
        return dfs(node.left,feat)
    else:
        return dfs(node.right,feat)

class LeastSqtTree(object):
    def __init__(self, feat, label, ep=0.2):
        self.feat = feat
        self.label = label
        self.root = None
        self.ep = ep
    
    def Train(self):
        self.root = build(self.feat,self.label,self.ep)
    
    def predict(self, feat):
        return dfs(self.root,feat)

train_X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
y = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00])
lst = LeastSqtTree(train_X, y)
lst.Train()
test_X = np.array([[2,4,5.5,6.5,7.5,9]]).T
for x in test_X:
    print(lst.predict(x))

