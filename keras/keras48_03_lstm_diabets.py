from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np


##1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)  # (309, 10) (133, 10)

x_train = x_train.reshape(309, 5, 2)
x_test = x_test.reshape(133, 5, 2)


##2. 모델구성
model = Sequential()
model.add(LSTM(40, activation ='relu', input_shape=(5, 2)))
model.add(Dropout(0.3))
model.add(Dense(40, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=10, verbose=3)


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
결과
loss: 40.88053894042969
RMSE: 50.900426549029675
R2: 0.5311752595365334
'''