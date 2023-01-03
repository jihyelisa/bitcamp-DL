import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

##1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
# 입력값과 결과값의 개수는 다를 수 있다

x, y = x.T, y.T
print(x.shape, y.shape)   #(10, 3) (10, 2)

##2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=3))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(2))

##3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=3000, batch_size=1)

##4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict([[9, 30, 210]])

print('Loss is', loss)
print('Result is', result)


'''
모델
model.add(Dense(50, input_dim=3))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(2))

훈련
model.fit(x, y, epochs=3000, batch_size=1)

결과
Loss is 0.09211696684360504
Result is [[10.034101   1.5485922]]
'''