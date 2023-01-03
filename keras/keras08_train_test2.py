import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
##1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])   #(10, )
y = np.array(range(10))   #(10, )

# x_train = x[:7]
# x_test = x[7:]
# y_train = y[:7]
# y_test = y[7:]
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    # test_size=0.3,
    # shuffle=True,
    random_state=123)
print(x_train, x_test, y_train, y_test)
#전체에서 골라내 훈련/평가로 나누면 과적합 문제가 생길 수 있음
#한쪽으로 어느정도 치우치는 랜덤값으로 하는 것이 더 나음

'''

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
model.fit(x_train, y_train, epochs=10, batch_size=1)

##4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([11])
print('loss:', loss)
print('result:', result)

'''