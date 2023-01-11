from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


##1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset['target']

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
# print('최소값:', np.min(x))  # 최소값: 0.0
# print('최대값:', np.max(x))  # 최대값: 1.0

print(type(x))  #  <class 'numpy.ndarray'>

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=20)


##2. 모델구성
model = Sequential()
model.add(Dense(40, activation ='relu', input_dim=13))
model.add(Dense(50, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(60, activation ='relu'))
model.add(Dense(60, activation ='relu'))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_split=0.2, verbose=3)


##4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse, ' / mae:', mae)

y_predict = model.predict(x_test)
# print(y_test)
# print(y_predict)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2:", r2)


'''
# validation, Scaler 사용 전

random_state=20

##2. 모델구성
model = Sequential()
model.add(Dense(40, activation ='relu', input_dim=13))
model.add(Dense(50, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(60, activation ='relu'))
model.add(Dense(60, activation ='relu'))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1)


loss: 2.860525608062744
RMSE: 4.038535734922319
R2: 0.8088525708349656

'''


'''
# validation, MinMaxScaler 사용 후

mse: 9.862133979797363  / mae: 2.220327377319336
RMSE: 3.1404035174180116
R2: 0.884417652338522
'''