import numpy as np
from sklearn.datasets import load_wine

##1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape) # (178, 13) (178,)

print(np.unique(y))
# [0 1 2] y값 label에 어떤 것이 있는지

print(np.unique(y, return_counts=True))
# y값이 label 별로 각각 몇 개인지
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))


# 힘들어서 몬하겟다...
# iris 데이터처럼 다중분류 하면 됨