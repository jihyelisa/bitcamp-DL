import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

##1. 데이터
x = np.array(range(10))
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])
# 입력값과 결과값의 개수는 다를 수 있다

y = y.T
print(x.shape, y.shape)   #(10, ) (10, 3)

##2. 모델구성
model = Sequential()
model.add(Dense(80, input_dim=1))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(3))

##3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=1500, batch_size=1)

##4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict([[9]])

print('Loss is', loss)
print('Result is', result)


'''
모델
model.add(Dense(80, input_dim=1))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(3))

훈련
model.fit(x, y, epochs=1500, batch_size=1)

결과
Loss is 0.07745935022830963
Result is [[ 9.980939    1.6718165  -0.07960853]]
'''