import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



##1. 데이터
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# index_col=0 : 데이터에서 제외할 인덱스 칼럼이 0열에 있음
submission = pd.read_csv(path + 'submission.csv', index_col=0)

# print(train_csv.shape)   # (1459, 10)

print(train_csv.isnull().sum())   # 열 별 결측치 보여줌
train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1)   #칼럼의 축 axis
# print(x)   # [1459 rows x 9 columns]
y = train_csv['count']
# print(y.shape)   # (1459,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.9,
    random_state=209)


##2. 모델구성
model = Sequential()
model.add(Dense(50, activation="relu", input_dim=9))
model.add(Dense(50, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=10,
          validation_split=0.125)


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
submission.to_csv(path + 'submission_0106_.csv')



'''
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.9,
    random_state=209)


##2. 모델구성
model = Sequential()
model.add(Dense(50, activation="relu", input_dim=9))
model.add(Dense(50, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=10,
          validation_split=0.125)

결과
loss: 2300.5771484375
RMSE: 47.96433290910494
R2: 0.6904905666724571
'''