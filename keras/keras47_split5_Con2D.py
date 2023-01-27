import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout


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

# Con2D에 넣기 위해 4차원으로 reshape 해준다.
# 흑백 이미지처럼 마지막에 1을 추가해줌
x_train = x_train.reshape(86, 2, 2, 1)
x_test = x_test.reshape(10, 2, 2, 1)
# print(x_train.shape, x_test.shape)  # (86, 2, 2, 1) (10, 2, 2, 1)


##2. 모델구성
model = Sequential()
model.add(Conv2D(160, (1,1), activation='relu', input_shape=(2, 2, 1)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(96, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=2, epochs=150, verbose=3)


##4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

x_predict = split_x(x_predict, 4) 
# x_predict는 x, y값으로 나눌 필요 없음.
# input할 shape인 숫자 4개 덩어리로만 맞춰준다.

# print(x_predict.shape)  # (7, 4)
# input 데이터 x와 같이 4차원으로 reshape 해준다.
x_predict = x_predict.reshape(7, 2, 2, 1)

result = model.predict(x_predict)
print('\nx_predict의 예측 결과:\n', result)


'''
##2. 모델구성
model = Sequential()
model.add(Conv2D(160, (1,1), activation='relu', input_shape=(2, 2, 1)))
model.add(Conv2D(128, (1,1), activation='relu'))
model.add(Flatten())
# model.add(Dropout(0.25))
model.add(Dense(96, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=3)


결과
loss: 7.648823157069273e-07

x_predict의 예측 결과:
 [[ 99.99862 ]
 [100.9986  ]
 [101.99859 ]
 [102.99856 ]
 [103.998535]
 [104.99853 ]
 [105.99851 ]]
'''