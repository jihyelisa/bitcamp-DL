import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, concatenate, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


##1. 데이터 전처리

###1-1. csv 불러오기

"""
# colab에서 돌릴 때
from google.colab import drive
path = '/content/drive/MyDrive/stock_contest/'
samsung_csv = pd.read_csv(path + '삼성전자 주가.csv',
                          index_col=0, header=0, encoding='cp949', thousands=',', nrows=1166)
amore_csv = pd.read_csv(path + '아모레퍼시픽 주가.csv',
                        index_col=0, header=0, encoding='cp949', thousands=',', nrows=1166)
"""

samsung_csv = pd.read_csv('삼성전자 주가.csv', index_col=0, header=0,
                          encoding='cp949', thousands=',', nrows=1166)
amore_csv = pd.read_csv('아모레퍼시픽 주가.csv', index_col=0, header=0,
                        encoding='cp949', thousands=',', nrows=1166)


#################################################################################
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc0 in position 0
# 디코드 에러 해결을 위해 encoding 파라미터 추가
#################################################################################
# 천 단위 구분자 ','가 포함된 컬럼을 int 형태로 바꾸기 위해 thousands 파라미터 추가
# print(samsung_csv.dtypes, amore_csv.dtypes)
#################################################################################
# ValueError: Data cardinality is ambiguous: Make sure all arrays contain the same number of samples.
# 아모레 데이터의 ValueError 해결을 위해 nrows 파라미터 추가해 삼성과 같은 삼전과 같은 행 수만 불러옴

samsung_csv = samsung_csv.sort_index()  #날짜 기준 순차 정렬
amore_csv = amore_csv.sort_index()
# print(samsung_csv[:10], amore_csv[:10])



###1-2. 사용할 칼럼만 추출
samsung = samsung_csv[['시가', '고가', '저가', '종가', '거래량']]
amore = amore_csv[['시가', '고가', '저가', '종가', '거래량']]
# print(samsung[:3], amore[:3])

samsung = samsung.values  # pandas 데이터를 numpy array로 변경
amore = amore.values
# print(type(samsung), type(amore))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# print(samsung.shape, amore.shape)
# (1166, 5) (1166, 5)



###1-3. split 함수로 x, y 나누기
def split_by5(dataset, timesteps, y_column):
    x, y = [], []
    for i in range(len(dataset)):
        x_end_number = i + timesteps
        y_end_number = x_end_number + y_column
        
        if y_end_number > len(dataset):  # 데이터 끝까지 가면 함수 자동 종료
            break
        
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

sam_x, sam_y = split_by5(samsung, 5, 1)
amo_x, amo_y = split_by5(amore, 5, 1)
# print(sam_x[:2, :], sam_x.shape)  # (1161, 5, 5)
# print(sam_y[:2], sam_y.shape)  # (1161, 1)
# print(amo_x[:2, :], amo_y.shape)  # (1161, 5, 5)
# print(amo_y[:2], amo_y.shape)  # (1161, 1)
"""
# LSTM에 적용하기 위해 reshape 해줌
sam_x = sam_x.reshape(1161, 25, 1)
amo_x = amo_x.reshape(1161, 25, 1)
# print(sam_x.shape, amo_x.shape)  # (1161, 25, 1) (1161, 25, 1)
"""


###1-4. train_test_split
sam_x_train, sam_x_test, amo_x_train, amo_x_test, sam_y_train, sam_y_test = train_test_split(
    sam_x, amo_x, sam_y, train_size=0.8, random_state=314)
print(sam_x_train.shape, sam_x_test.shape)  # (928, 5, 5) (233, 5, 5)
print(amo_x_train.shape, amo_x_test.shape)  # (928, 5, 5) (233, 5, 5)
print(sam_y_train.shape, sam_y_test.shape)  # (928, 1) (233, 1)




###1-5. scaler 사용하기
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler1.fit(sam_x_train)
scaler2.fit(amo_x_train)
# scaler = StandardScaler()

sam_x_train = sam_x_train.reshape(928, 25)
sam_x_test = sam_x_test.reshape(233, 25)
amo_x_train = amo_x_train.reshape(928, 25)
amo_x_test = amo_x_test.reshape(233, 25)

sam_x_train = scaler1.transform(sam_x_train)
sam_x_test = scaler1.transform(sam_x_test)
amo_x_train = scaler2.transform(amo_x_train)
amo_x_test = scaler2.transform(amo_x_test)

sam_x_train = sam_x_train.reshape(928, 25, 1)
sam_x_test = sam_x_test.reshape(233, 25, 1)
amo_x_train = amo_x_train.reshape(928, 25, 1)
amo_x_test = amo_x_test.reshape(233, 25, 1)



##2. 모델구성

###2-1. 삼성전자 모델
input1 = Input(shape=(25,))
lstm1 = Bidirectional(LSTM(72, activation='relu', name='dns11'))(input1)
# drop1 = Dropout(0.2)(lstm1)
dense2 = Dense(60, activation='relu', name='dns12')(lstm1)
dense3 = Dense(48, activation='relu', name='dns13')(dense2)
output1 = Dense(30, activation='relu', name='dns14')(dense3)


###2-2. 아모레퍼시픽 모델
input2 = Input(shape=(25,))
lstm21 = Bidirectional(GRU(50, activation='relu', name='dns21'))(input2)
# drop21 = Dropout(0.2)(lstm21)
dense22 = Dense(70, activation='relu', name='dns22')(lstm21)
dense23 = Dense(50, activation='relu', name='dns23')(dense22)
output2 = Dense(20, activation='relu', name='out2')(dense23)


###2-3. 모델병합
merge1 = concatenate([output1, output2], name='mrg1')
merge2 = Dense(16, activation='relu', name='mrg2')(merge1)
merge3 = Dense(12, activation='relu', name='mrg3')(merge2)
last_output = Dense(1, name='last')(merge3)  # sam_y 데이터 칼럼이 1개이므로 최종 output은 1

model = Model(inputs=[input1, input2], outputs=last_output)


##3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor="val_loss",
                              mode="min", patience=30, restore_best_weights=True)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")  # 현재 날짜와 시간
filepath = './save_MCP_scaler/'
filename = '{epoch:03d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf5

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                      filepath=filepath + 'samsung_' + date + '_' + filename)

model.fit([sam_x_train, amo_x_train], sam_y_train, epochs=200, batch_size=20,
          validation_split=0.1, callbacks=[es, mcp], verbose=3)


##### 저장한 모델 불러오기 #####
# model = load_model('save_MCP/samsung_0129_1621_065-861947.3750.hdf5')


##4. 평가, 예측

loss = model.evaluate([sam_x_test, amo_x_test], sam_y_test)

sam_x = sam_x.reshape(1161, 25)
amo_x = amo_x.reshape(1161, 25)

result = model.predict([sam_x, amo_x])
print('loss:', loss)
print('mse:')
for i in range(3, 0, -1):
  i *= -1
  print(f'2023.01.2{i + 8} 실제 시가: {sam_y[i]} / 예측 시가: {result[i]}')
print('2023.01.30 삼성전자 예측 시가:', result[-1])
''''''

"""
"""