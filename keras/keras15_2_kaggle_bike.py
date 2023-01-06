import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


##1. 데이터
path = './_data/bike/'
# 변수에 데이터파일 할당
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

# print(train_csv.shape, test_csv.shape)   # (10886, 11) (6493, 8)
# print(train_csv.columns, test_csv.columns)   # 없는 칼럼 조회
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   # 없는 칼럼 제외한 input
y = train_csv['count']   # output
# print(x.shape, y.shape)   # (10886, 8) (10886,)

# print(train_csv.isnull().sum(), test_csv.isnull().sum())   # 결측치 없음

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7,
    random_state=341)


##2. 모델구성
model = Sequential()
model.add(Dense(40, activation ='relu', input_dim=8))
model.add(Dense(50, activation ='relu'))
model.add(Dense(60, activation ='relu'))
model.add(Dense(70, activation ='relu'))
model.add(Dense(80, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(30, activation ='relu'))
model.add(Dense(1))

##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# start = time.time()
model.fit(x_train, y_train, epochs=125, batch_size=40)
# end = time.time()
# print("train-time:", end - start)   # 훈련 소요시간 측정


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
submission.to_csv(path + 'mySubmission_0106.csv')


'''
train time record
cpu: 40.9
gpu: no need to see
'''


'''
random_state=341

##2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=8))
model.add(Dense(50, activation ='relu'))
model.add(Dense(60, activation ='relu'))
model.add(Dense(70, activation ='relu'))
model.add(Dense(80, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(30, activation ='relu'))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# start = time.time()
model.fit(x_train, y_train, epochs=150, batch_size=40)

result
loss: 21478.228515625
RMSE: 146.55452819238474
R2: 0.32248623070217464
'''