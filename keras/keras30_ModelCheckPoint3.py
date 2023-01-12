from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
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


''''''
##2. 함수형 모델구성


input1 = Input(shape=(13,))
dense1 = Dense(100, activation='relu')(input1)
dense2 = Dense(80, activation ='relu')(dense1)
dense3 = Dense(60, activation ='relu')(dense2)
dense4 = Dense(40, activation ='relu')(dense3)
dense5 = Dense(20, activation ='relu')(dense4)
output1 = Dense(1, activation ='relu')(dense5)
model = Model(inputs=input1, outputs=output1)


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor="val_loss", mode="min",
                              patience=10,
                              restore_best_weights=True
                            )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=1, save_best_only=True,
                      filepath=path + 'MCP/keras30_ModelCheckPoint3.hdf5')

model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_split=0.125, callbacks=[es, mcp], verbose=3)

model.save(path + 'keras30_ModelCheckPoint3_save_model.h5')


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


print("================== 2. load_model 출력 =================")

model2 = load_model(path + 'keras30_ModelCheckPoint3_save_model.h5')

mse, mae = model2.evaluate(x_test, y_test)
print('mse:', mse, ' / mae:', mae)

y_predict = model2.predict(x_test)

print("RMSE:", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2:", r2)


print("================== 3. ModelCheckPoint 출력 =================")

model3 = load_model(path + 'MCP/keras30_ModelCheckPoint3.hdf5')

mse, mae = model3.evaluate(x_test, y_test)
y_predict = model3.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('mse:', mse, ' / mae:', mae)
print("RMSE:", RMSE(y_test, y_predict))
print("R2:", r2)