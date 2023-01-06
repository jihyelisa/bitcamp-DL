import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


##1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.625, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=0.5, random_state=10)


##2. 모델
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=1))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))

##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))

##4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

result = model.predict([17])
print('17의 예측값:', result)


'''
random_state=10


##2. 모델
model = Sequential()
model.add(Dense(30, input_dim=1))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=80, batch_size=1,
          validation_data=(x_val, y_val))

loss: 0.00027992032119072974
17의 예측값: [[16.986368]]
'''