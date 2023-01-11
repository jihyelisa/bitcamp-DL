# [과제, 실습]
# R2 0.62~ 이상

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


##1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target



x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7,
    random_state=9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


##2. 모델구성
model = Sequential()
model.add(Dense(100, activation="relu", input_dim=10))
model.add(Dense(70, activation="relu"))
model.add(Dense(40, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))



##3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=30, verbose=3)


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
model.add(Dense(100, activation="relu", input_dim=10))
model.add(Dense(70, activation="relu"))
model.add(Dense(40, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=80, batch_size=60)


# scaler 사용 전
loss: 40.75837707519531
RMSE: 49.662574365679795
R2: 0.5537007725161254

# scaler 사용 후
loss: 37.43274688720703
RMSE: 47.028249000122166
R2: 0.5997924326058346
'''