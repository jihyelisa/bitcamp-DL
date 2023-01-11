
## 일직선 모양이 아니라서 Sequential 모델로 처리할 수 없는 경우 함수 모델을 사용

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


##1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=20)


scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



##2. 모델구성(순차형)
model = Sequential()
model.add(Dense(50, activation ='relu', input_shape=(13,)))
model.add(Dense(40, activation ='sigmoid'))
model.add(Dense(30, activation ='relu'))
model.add(Dense(20))
model.add(Dense(1))


##2. 모델구성(함수형) - 위와 코드구성만 다를 뿐 동일한 과정을 거친다
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation ='sigmoid')(dense1)
dense3 = Dense(30, activation ='relu')(dense2)
dense4 = Dense(20)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)

'''
Model: "sequential" 순차형
_________________________________________________________________        
 Layer (type)                Output Shape              Param #
=================================================================        
 dense (Dense)               (None, 50)                700

 dense_1 (Dense)             (None, 40)                2040

 dense_2 (Dense)             (None, 30)                1230

 dense_3 (Dense)             (None, 20)                620

 dense_4 (Dense)             (None, 1)                 21

=================================================================        
Total params: 4,611
Trainable params: 4,611
Non-trainable params: 0
_________________________________________________________________        
None
'''

'''
Model: "model" 함수형
_________________________________________________________________        
 Layer (type)                Output Shape              Param #
=================================================================        
 input_1 (InputLayer)        [(None, 13)]              0

 dense (Dense)               (None, 50)                700

 dense_1 (Dense)             (None, 40)                2040      

 dense_2 (Dense)             (None, 30)                1230

 dense_3 (Dense)             (None, 20)                620

 dense_4 (Dense)             (None, 1)                 21

=================================================================        
Total params: 4,611
Trainable params: 4,611
Non-trainable params: 0
_________________________________________________________________
'''

print(model.summary())



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




# scaler 사용 전
loss: 2.860525608062744
RMSE: 4.038535734922319
R2: 0.8088525708349656

# scaler 사용 후
mse: 12.514263153076172  / mae: 2.400543689727783
RMSE: 3.5375504819219334
R2: 0.8533352003289097

'''