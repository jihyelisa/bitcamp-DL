from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


##1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])


##2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()  # 모델의 아키텍처를 출력해줌

'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape 모델의 구조   Param # 연산량
=================================================================
 dense (Dense)               (None, 5)                 10        

 dense_1 (Dense)             (None, 4)                 24        

 dense_2 (Dense)             (None, 3)                 15

 dense_3 (Dense)             (None, 2)                 8

 dense_4 (Dense)             (None, 1)                 3

=================================================================
Total params: 60
Trainable params: 60
Non-trainable params: 0
_________________________________________________________________

# 왜 연산량이 아래 노드 개수만큼 한 번 더 더해질까?
# bias 값을 더하는 과정이 한번 추가됨
'''



