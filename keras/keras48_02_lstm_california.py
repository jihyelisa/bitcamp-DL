# 31_2 파일 가져오기

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np
import time


##1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=9)


from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (14447, 8) (6193, 8) (14447,) (6193,)

x_train = x_train.reshape(14447, 4, 2)
x_test = x_test.reshape(6193, 4, 2)

print(x_train.shape, x_test.shape)  # (14447, 2, 2) (6193, 2, 2)



##2. 모델구성
model = Sequential()
model.add(LSTM(70, activation ='relu', input_shape=(4, 2)))
model.add(Dropout(0.3))
model.add(Dense(60, activation ='relu'))
model.add(Dropout(0.2))
model.add(Dense(70, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(30, activation ='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=20,
          validation_split=0.2, verbose=3)


##4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2:", r2)

'''
loss: 0.42635974287986755
RMSE: 0.6434235720501599
R2: 0.6901512366753195
'''