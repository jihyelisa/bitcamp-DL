import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

##1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])

# [실습] train_test_split을 이용하여
# 7:3으로 잘라서 모델 구현 / 소스 완성

x, y = x.T, y.T   #(10, 3) (10, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7, 
    random_state=5
)

# print(x_train, x_test, y_train, y_test)

model = Sequential()
model.add(Dense(100, input_dim=3))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(2))

model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1)

loss = model.evaluate(x_test, y_test)
result = model.predict([9, 30, 210])

print('Loss is', loss)
print('Result is', result)


