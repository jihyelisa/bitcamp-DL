from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time


##1. 데이터
datasets = fetch_california_housing()
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


'''
##2. 모델구성
model = Sequential()
model.add(Dense(40, activation ='relu', input_dim=8))
model.add(Dense(50, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(60, activation ='relu'))
model.add(Dense(60, activation ='relu'))
model.add(Dense(70, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(30, activation ='relu'))
model.add(Dense(10, activation ='relu'))
model.add(Dense(1))
'''

##2. 함수형 모델구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout

input1 = Input(shape=(8,))
dense1 = Dense(40, activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(50, activation ='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(50, activation ='relu')(drop2)
dense4 = Dense(50, activation ='relu')(dense3)
dense5 = Dense(60, activation ='relu')(dense4)
dense6 = Dense(60, activation ='relu')(dense5)
dense7 = Dense(70, activation ='relu')(dense6)
dense8 = Dense(50, activation ='relu')(dense7)
dense9 = Dense(30, activation ='relu')(dense8)
dense10 = Dense(10, activation ='relu')(dense9)
output1 = Dense(1)(dense10)
model = Model(inputs=input1, outputs=output1)


##3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=30,
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