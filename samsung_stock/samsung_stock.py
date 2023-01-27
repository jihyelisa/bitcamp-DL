import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split


##1. 데이터

###1-1. 데이터 전처리


###1-2.
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

train_csv = train_csv.dropna()
x = train_csv.drop(['count'], axis=1)   #칼럼의 축 axis
y = train_csv['count']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=209)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
# fit은 기준을 정하는 느낌?? 한 번만 해준다
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)  # (1195, 9) (133, 9)
x_train = x_train.reshape(1195, 3, 3)
x_test = x_test.reshape(133, 3, 3)



'''
x1_datasets = np.array([range(100), range(301, 401)]).transpose()  # transpose: 전치
# print(x1_datasets.shape)  # (100, 2)  # 삼전 시가, 고가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose()
# print(x2_datasets.shape)  # (100, 3)  # 아모레 시가, 고가, 종가

y = np.array(range(2001, 2101))  #(100,)  # 삼전의 하루 뒤 종가

from sklearn.model_selection import train_test_split
# 3개 이상의 데이터셋도 train_test_split로 분리 가능
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y, train_size=0.7, random_state=123)

# print(x1_train.shape, x2_train.shape, y_train.shape)  # (70, 2) (70, 3) (70,)
# print(x1_test.shape, x2_test.shape, y_test.shape)  # (30, 2) (30, 3) (30,)


##2. 모델구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

###2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='dns11')(input1)
dense2 = Dense(12, activation='relu', name='dns12')(dense1)
dense3 = Dense(13, activation='relu', name='dns13')(dense2)
output1 = Dense(14, activation='relu', name='dns14')(dense3)

###2-2. 모델2
input2 = Input(shape=(3,))
dense21 = Dense(21, activation='linear', name='dns21')(input2)
dense22 = Dense(22, activation='linear', name='dns22')(dense21)
output2 = Dense(23, activation='linear', name='dns23')(dense22)

###2-3. 모델병합
from tensorflow.keras.layers import concatenate 
# concatenate: 사슬처럼 엮다, 단순히 이음, 병합시킴
merge1 = concatenate([output1, output2], name='mrg1')
merge2 = Dense(12, activation='relu', name='mrg2')(merge1)
merge3 = Dense(13, activation='relu', name='mrg3')(merge2)
last_output = Dense(1, name='last')(merge3)  # y 데이터 칼럼이 1개이므로 최종 output은 1

model = Model(inputs=[input1, input2], outputs=last_output)
# 함수형 모델은 마지막에 모델 정의
# 모델의 시작과 끝을 명시해줌

# model.summary()


##3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=200, batch_size=2)


##4. 평가, 예측

loss = model.evaluate([x1_test, x2_test], y_test)
print('loss:', loss)
'''