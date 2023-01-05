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
train_csv, test_csv = train_csv.dropna(axis=0), test_csv.dropna(axis=0)
# index_col=0 : 데이터에서 제외할 인덱스 칼럼이 0열에 있음
submission = pd.read_csv(path + 'submission.csv', index_col=0)

# print(train_csv.shape)   # (1459, 10)
# # id 제외해야 하므로 input_dim=9

# print(train_csv.columns)
# # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
# #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',        
# #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
# #       dtype='object')

# print(train_csv.info())
# #  #   Column                  Non-Null Count  Dtype
# # ---  ------                  --------------  -----
# #  0   hour                    1459 non-null   int64
# #  1   hour_bef_temperature    1457 non-null   float64     #결측치 2
# #  2   hour_bef_precipitation  1457 non-null   float64     #결측치 2
# #  3   hour_bef_windspeed      1450 non-null   float64     #결측치 9
# #  4   hour_bef_humidity       1457 non-null   float64        .
# #  5   hour_bef_visibility     1457 non-null   float64        .
# #  6   hour_bef_ozone          1383 non-null   float64        .
# #  7   hour_bef_pm10           1369 non-null   float64        .
# #  8   hour_bef_pm2.5          1342 non-null   float64        .
# #  9   count                   1459 non-null   float64        .

# # 결측치가 있는 데이터를 어떻게 처리할까?
# # 1. 삭제해서 사용하지 않는다.

# # print(test_csv.info())
# # print(train_csv.describe())

x = train_csv.drop(['count'], axis=1)   #칼럼의 축 axis
# print(x)   # [1459 rows x 9 columns]
y = train_csv['count']
# print(y)
# print(y.shape)   # (1459,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7,
    random_state=209)



##2. 모델구성
model = Sequential()
model.add(Dense(300, activation="relu", input_dim=9))
model.add(Dense(200, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=1)



##4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))



##5. 제출 파일
# y_submit = model.predict(test_csv)