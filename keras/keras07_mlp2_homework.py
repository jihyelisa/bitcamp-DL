import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


##1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape, y.shape)   #(3, 10) (10,)
x = x.T
print(x.shape)   #(10, 3)


##2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=3))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


##3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)


##4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict([[10,1.4,0]])
print('Loss is', loss)
print('Result is', result)

'''
결과
Loss is 0.04883342981338501
Result is [[19.904491]]
'''