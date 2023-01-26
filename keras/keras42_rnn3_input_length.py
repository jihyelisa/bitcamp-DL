import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


##1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])  # (10,)

# y = ???   # y 데이터가 따로 제공되지 않으므로 데이터셋을 잘라 x, y를 만든다.
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])  #(7, 3)
y = np.array([4,5,6,7,8,9,10])  #(7,)

x = x.reshape(7, 3, 1)  # 데이터 덩어리 안에서 한 칸씩 이동하며 예측하겠다는 의미


##2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=160, activation='relu', input_shape=(3, 1)))
model.add(SimpleRNN(units=160, input_length=3, input_dim=1))
                            # 칼럼 값을 input_length에 넣을 수도 있음 (똑같은 의미임!)
                            # 가독성이 떨어져서 잘 사용하지 않음

model.add(Dense(128, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
