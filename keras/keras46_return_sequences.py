import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

##1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5],
              [4,5,6], [5,6,7], [6,7,8],
              [7,8,9], [8,9,10], [9,10,11],
              [10,11,12], [20,30,40],
              [30,40,50], [40,50,60]])  #(13, 3)
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])  #(13,)
x_predict = np.array([50,60,70])

x = x.reshape(13, 3, 1)


##2. 모델구성
model = Sequential()
model.add(LSTM(160, activation='relu', input_shape=(3, 1),
               return_sequences=True))  # 다음 input_shape를 3차원으로 맞춰줌
model.add(LSTM(32))  # input_shape (N, 160)
model.add(Dense(128, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=150, verbose=3)


##4. 평가, 예측
loss = model.evaluate(x, y)
print('loss:', loss)
y_pred = np.array(x_predict)  # model의 input_shape와 맞지 않아((3,)) 에러
y_pred = y_pred.reshape(1, 3, 1)  # input 데이터 x와 같이 3차원으로 reshape 해준다
result = model.predict(y_pred)
print('[50,60,70]의 예측 결과:', result)


'''
##2. 모델구성
model = Sequential()
model.add(LSTM(160, activation='relu', input_shape=(3, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=150, verbose=3)


//result//
loss: 0.08889644593000412
[50,60,70]의 예측 결과: [[80.465546]]
'''