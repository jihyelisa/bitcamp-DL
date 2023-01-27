# 47_2 가져옴

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional


##1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))
# 예상 y = 100 ~ 106

timesteps = 5  # x는 4개, y는 1개


# 실습

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
    x, y, shuffle=True, train_size=0.9, random_state=3
)

# print(x_train.shape, x_test.shape)  # (86, 4) (10, 4)

x_train = x_train.reshape(86, 4, 1)
x_test = x_test.reshape(10, 4, 1)
# print(x_train.shape, x_test.shape)  # (86, 4, 1) (10, 4, 1)


##2. 모델구성
model = Sequential()
# 순방향으로만 훈련시키던 것을 반대 방향으로도 실행
# 데이터 증폭 없이 두 배 훈련시킬 수 있다
# 주의점) Bidirectional은 방향만 정해주는 것으로, 훈련 방식을 정해주어야 한다.
model.add(Bidirectional(LSTM(160, activation='relu'), input_shape=(4, 1)))
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
x_predict = x_predict.reshape(7, 4, 1)  # input 데이터 x와 같이 3차원으로 reshape 해준다.

result = model.predict(x_predict)
print('x_predict의 예측 결과:\n', result)


'''
##2. 모델구성
model = Sequential()
# Bidirectional은 방향만 정해주는 것으로, 훈련 방식을 정해주어야 한다.
model.add(Bidirectional(LSTM(160, activation='relu'), input_shape=(4, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=3)


결과
loss: 0.0007399669848382473
x_predict의 예측 결과:
 [[100.00367]
 [101.00369]
 [102.00372]
 [103.0037 ]
 [104.00365]
 [105.00358]
 [106.00347]]
'''