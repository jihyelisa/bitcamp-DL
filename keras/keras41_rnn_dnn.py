import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


##1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])  # (10,)

# y = ???   # y 데이터가 따로 제공되지 않으므로 데이터셋을 잘라 x, y를 만든다.
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])  #(7, 3)
y = np.array([4,5,6,7,8,9,10])  #(7,)

# x = x.reshape(7, 3, 1)  # 데이터 덩어리 안에서 한 칸씩 이동하며 예측하겠다는 의미


##2. 모델구성
model = Sequential()
# model.add(SimpleRNN(128, activation='relu', input_shape=(3, 1)))
model.add(Dense(128, activation='relu', input_shape=(3,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=200)


##4. 평가, 예측
loss = model.evaluate(x, y)
print(loss)
y_pred = np.array([8,9,10])  # model의 input_shape와 맞지 않아((3,)) 에러
y_pred = y_pred.reshape(1, 3, 1)  # input 데이터 x와 같이 3차원으로 reshape 해준다
result = model.predict(y_pred)
print('[8,9,10]의 예측 결과:', result)



'''
##2. 모델구성
model = Sequential()
model.add(SimpleRNN(128, activation='relu', input_shape=(3, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=200)



결과
4.6647755880258046e-08
[8,9,10]의 예측 결과: [[11.02034]]
'''