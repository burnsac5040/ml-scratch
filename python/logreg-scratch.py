#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
# %matplotlib inline
plt.style.use(['dark_background'])


# In[2]:


dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

idx = list(np.arange(X.shape[0]))
np.random.shuffle(idx)

train_idx, test_idx = np.split(idx, [int(0.8*len(idx))])

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print('X_train: %s, X_test %s\ny_train: %s, y_test: %s' % (X_train.shape, X_test.shape, y_train.shape, y_test.shape))


# In[3]:


from timeit import default_timer as t

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000, bias=True):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = bias

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X, y):
        if isinstance(self.b, bool):
            X = np.hstack([np.ones([X.shape[0], 1]), X])

        n_samp, n_feat = X.shape
        self.w = np.random.rand(n_feat)
        
        start = t()
        for _ in range(self.n_iters):
            z = np.dot(X, self.w)
            y_pred = self._sigmoid(z)
            dw = (1 / n_samp) * np.dot(X.T, (y_pred - y))
            self.w -= self.lr * dw
            
        end = t()
        print(f'Elapsed time: {end - start:.4f}')
            
    def predict_proba(self, X, threshold=0.5):
        if isinstance(self.b, bool):
            X = np.hstack([np.ones([X.shape[0], 1]), X])

        probs = self._sigmoid(np.dot(X, self.w))
        clss = probs >= threshold
        return probs, clss


# In[4]:


lr = LogisticRegression()
lr.fit(X_train, y_train)
_, pred = lr.predict_proba(X_test)

print(f'Accuracy: {np.sum(y_test == pred.astype(int)) / y_test.size}')

