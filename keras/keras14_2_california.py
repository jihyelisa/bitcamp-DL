# [실습]
# R2 0.55~0.6 이상

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


##1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x, y)
# print(x.shape, y.shape)   # (20640, 8) (20640, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7,
    random_state=9)


##2. 모델구성
model = Sequential()
model.add(Dense(40, activation ='ReLU', input_dim=8))
model.add(Dense(50, activation ='ReLU'))
model.add(Dense(50, activation ='ReLU'))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=30)



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
random_state=9


##2. 모델구성
model = Sequential()
model.add(Dense(40, activation ='ReLU', input_dim=8))
model.add(Dense(50, activation ='ReLU'))
model.add(Dense(50, activation ='ReLU'))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=30)


결과
loss: 0.46151232719421387
RMSE: 0.6640760671653969
R2: 0.6699410680008235
'''