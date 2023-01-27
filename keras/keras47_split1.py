import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


a = np.array(range(1, 11))
timesteps = 5

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape)  #(6, 5)

x = bbb[:, :-1]
y = bbb[:, -1]  # numpy 배열 slicing
# print(x, y)
print(x.shape, y.shape)  # (6, 4) (6,)
x = x.reshape(6, 4, 1)
print(x.shape, y.shape)  # (6, 4, 1) (6,)

x_predict = np.array([7,8,9,10])


# 실습
# LSTM 모델 구성

##2. 모델구성
model = Sequential()
model.add(LSTM(160, activation='relu', input_shape=(4, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=200, verbose=3)


##4. 평가, 예측
loss = model.evaluate(x, y)
print('loss:', loss)
x_predict = x_predict.reshape(1, 4, 1)  # input 데이터 x와 같이 3차원으로 reshape 해준다
result = model.predict(x_predict)
print('[7,8,9,10]의 예측 결과:', result)


'''
##2. 모델구성
model = Sequential()
model.add(LSTM(160, activation='relu', input_shape=(4, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=200, verbose=3)


## result
loss: 0.0018795226933434606
[7,8,9,10]의 예측 결과: [[11.002057]]
'''