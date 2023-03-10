import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [13] 예측해보기

##모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(1, input_dim=1))

##컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=5000)

##평가, 예측
result = model.predict([13])
print(f"result: {result}")