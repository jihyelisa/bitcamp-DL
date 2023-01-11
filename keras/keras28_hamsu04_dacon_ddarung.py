import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping



##1. 데이터
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

train_csv = train_csv.dropna()
x = train_csv.drop(['count'], axis=1)   #칼럼의 축 axis
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.9,
    random_state=209)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
# fit은 기준을 정하는 느낌?? 한 번만 해준다
x_test = scaler.transform(x_test)


'''
##2. 모델구성
model = Sequential()
model.add(Dense(50, activation="relu", input_dim=9))
model.add(Dense(50, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))
'''

##2. 함수형 모델구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(9,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(50, activation ='relu')(dense1)
dense3 = Dense(50, activation ='relu')(dense2)
dense4 = Dense(100, activation ='relu')(dense3)
dense5 = Dense(100, activation ='relu')(dense4)
dense6 = Dense(50, activation ='relu')(dense5)
output1 = Dense(1)(dense6)
model = Model(inputs=input1, outputs=output1)



##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

earlyStopping = EarlyStopping(monitor="val_loss",
                              mode="min", patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=200, batch_size=20,
          validation_split=0.1, callbacks=[earlyStopping], verbose=3)


##4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2:", r2)


##5. 제출 파일
test_csv = scaler.transform(test_csv)
# scaler 이용해 0~1 사이값으로 얻은 웨이트값이므로 제출파일 만들 때도 scaler 해준 뒤 예측한다

submission['count'] = model.predict(test_csv)
submission.to_csv(path + 'submission_0111_.csv')



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


# scaler 사용 전 결과
loss: 2300.5771484375
RMSE: 47.96433290910494
R2: 0.6904905666724571

# scaler 사용 후 결과
loss: 1770.7137451171875
RMSE: 42.079851207844456
R2: 0.7617760267378617
'''