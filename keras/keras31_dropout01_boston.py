from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
path = './_save/'


##1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=20)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


'''
##2. 순차형 모델구성
model = Sequential()
model.add(Dense(40, activation ='relu', input_dim=13))
model.add(Dropout(0.5))
model.add(Dense(50, activation ='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation ='relu'))
model.add(Dropout(0.2))
model.add(Dense(60, activation ='relu'))
model.add(Dense(60, activation ='relu'))
model.add(Dense(10))
model.add(Dense(1))
'''


##2. 함수형 모델구성

input1 = Input(shape=(13,))
dense1 = Dense(100, activation='relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(80, activation ='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(60, activation ='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(40, activation ='relu')(drop3)
drop4 = Dropout(0.1)(dense4)
dense5 = Dense(20, activation ='relu')(drop4)
output1 = Dense(1, activation ='relu')(dense5)
model = Model(inputs=input1, outputs=output1)


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor="val_loss", mode="min",
                              patience=10,
                              restore_best_weights=True)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")  # 현재 날짜와 시간

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf5


mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=1, save_best_only=True,
                      filepath=filepath + 'k31_01_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_split=0.125, callbacks=[es, mcp], verbose=3)

# model.save(path + 'keras30_ModelCheckPoint4_save_model.h5')


##4. 평가, 예측

print("================== 1. 기본 출력 =================")
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse, ' / mae:', mae)

y_predict = model.predict(x_test)


from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2:", r2)
