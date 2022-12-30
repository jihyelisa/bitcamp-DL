import tensorflow as tf
print(tf.__version__)
import numpy as np


##1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])


##2. 모델구성
from tensorflow.keras.models import Sequential   #순차적 모델
from tensorflow.keras.layers import Dense   #구성 형태
# tensorflow 안의 keras 안의 layers 안의 Dense를 가져옴

model = Sequential()
model.add(Dense(1, input_dim=1))   #dim: dimension
#1 : y값 행렬데이터 덩어리, input_dim=1 : x값 행렬데이터 덩어리
#"정제된" 데이터를 입력해야 함


##3-1. 컴파일(컴퓨터가 말을 알아먹게 한다!)
model.compile(loss='mae', optimizer='adam')   #참고) mae: minimum average error
#optimizer: loss를 최적화 해줌.
#adam: 사용했을 때 보통 85 정도 나옴

##3-2. 훈련
model.fit(x, y, epochs=2000)   #epochs: 훈련횟수


##4. 평가, 예측
result = model.predict([4])
print('결과: ', result)