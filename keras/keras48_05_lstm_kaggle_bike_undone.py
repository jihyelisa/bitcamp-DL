import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping



##1. 데이터
path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

# print(train_csv.shape, test_csv.shape)   # (10886, 11) (6493, 8)
# print(train_csv.columns, test_csv.columns)   # 없는 칼럼 조회
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   # 없는 칼럼 제외한 input
y = train_csv['count']   # output
# print(x.shape, y.shape)   # (10886, 8) (10886,)

x = x.reshape(10886, 4, 2)
test_csv = test_csv.reshape(6493, 4, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=341)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


##2. 모델구성
model = Sequential()
model.add(LSTM(units=50, activation ='relu', input_dim=(4, 2)))
model.add(Dense(50, activation ='relu'))
model.add(Dense(60, activation ='relu'))
model.add(Dense(70, activation ='relu'))
model.add(Dense(80, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(30, activation ='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

earlyStopping = EarlyStopping(monitor="val_loss",
                              mode="min", patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=300, batch_size=20, callbacks=[earlyStopping], verbose=3,
          validation_split=0.1)


##4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2:", r2)


#5. 제출 파일
submission['count'] = model.predict(test_csv)
submission.to_csv(path + 'mySubmission_0111_.csv')



'''
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.8,
    random_state=341)


##2. 모델구성
model = Sequential()
model.add(Dense(50, activation ='relu', input_dim=8))
model.add(Dense(50, activation ='relu'))
model.add(Dense(60, activation ='relu'))
model.add(Dense(70, activation ='relu'))
model.add(Dense(80, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(30, activation ='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=15,
          validation_split=0.125)

result
loss: 20286.9921875
RMSE: 142.43241996531088
R2: 0.3294971606927243
'''