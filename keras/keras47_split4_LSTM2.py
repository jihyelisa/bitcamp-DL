import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


##1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))
# 예상 y = 100 ~ 106

timesteps = 5  # x는 4개, y는 1개

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)
# print(bbb)
# print(bbb.shape)  #(96, 5)

x = bbb[:, :-1]
y = bbb[:, -1]
# print(x, y)
# print(x.shape, y.shape)  # (96, 4) (96,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.9, random_state=3)
# print(x_train.shape, x_test.shape)  # (86, 4) (10, 4)

# (batch, timesteps, feature)
# feature를 2로 하면,
# timesteps는 둘을 곱했을 때 x의 덩어리 크기인 4가 될 수 있게 2가 되어야 한다.
x_train = x_train.reshape(86, 2, 2)
x_test = x_test.reshape(10, 2, 2)
# print(x_train.shape, x_test.shape)  # (86, 2, 2) (10, 2, 2)


##2. 모델구성
model = Sequential()
model.add(LSTM(160, activation='relu', input_shape=(2, 2)))
model.add(Dense(128, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=3)


##4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

x_predict = split_x(x_predict, 4) 
# x_predict는 x, y값으로 나눌 필요 없음.
# input할 shape인 숫자 4개 덩어리로만 맞춰준다.

# print(x_predict.shape)  # (7, 4)
x_predict = x_predict.reshape(7, 2, 2)  # input 데이터 x와 같이 3차원으로 reshape 해준다.

result = model.predict(x_predict)
print('\nx_predict의 예측 결과:\n', result)


'''
##2. 모델구성
model = Sequential()
model.add(LSTM(160, activation='relu', input_shape=(2, 2)))
model.add(Dense(128, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=3)


결과
loss: 0.0002640957827679813

x_predict의 예측 결과:
 [[ 99.98688 ]
 [100.98705 ]
 [101.987236]
 [102.987465]
 [103.98772 ]
 [104.98797 ]
 [105.98826 ]]
'''