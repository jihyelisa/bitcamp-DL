import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

##1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])   #(10, )
# y = np.array(range(10))   #(10, )
x_train = np.array([1,2,3,4,5,6,7])   #(7, )
x_test = np.array([8,9,10])   #(3, )
y_train = np.array(range(7))   #(7, )
y_test = np.array(range(7, 10))   #(7, )


##2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(1))

##3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=3000, batch_size=1)

##4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([11])
print('loss:', loss)
print('result:', result)


'''
모델
model.add(Dense(100, input_dim=1))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(1))

훈련
model.fit(x_train, y_train, epochs=3000, batch_size=1)

결과
loss: 0.0019578933715820312
result: [[10.001951]]
'''


