import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential   #순차적 모델
from tensorflow.keras.layers import Dense   #구성 형태
	#tensorflow 안의 keras 안의 layers 안의 Dense를 가져옴



##1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])



##2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=1))
	#dim: dimension > 하나의 차원, 하나의 덩어리가 들어간다는 의미
	#1 : y값 행렬데이터 덩어리, input_dim=1 : x값 행렬데이터 덩어리
	#"정제된" 데이터를 입력해야 함
model.add(Dense(50))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
	#hidden layer이기 때문에 어떤 형태가 좋은지는 해봐야 앎
model.add(Dense(1))



##3-1. 컴파일(컴퓨터가 말을 알아먹게 한다!)
model.compile(loss='mae', optimizer='adam')
	#참고) mae: minimum average error
	#optimizer: loss를 최적화 해줌.
	#adam: 사용했을 때 보통 정확도 85% 정도 나옴

##3-2. 훈련
model.fit(x, y, epochs=10, batch_size=7)
	#epochs: 훈련횟수
	#batch_size 지정하지 않을 경우 default값 32
	#많이 훈련하면 최적의 w를 구할 수 있지만 지나치게 많이 하면 과적합될 수 있음



##4. 평가, 예측
loss = model.evaluate(x, y)
    #원래는 훈련데이터와 별개의 평가데이터로 평가함
    #model.evaluate(평가데이터) >>> loss값을 반환
print('loss:', loss)

result = model.predict([7])
print('7의 결과: ', result)

    #결과판단의 기준은 predict가 아닌 loss값
    #loss값이 작을수록 w값이 최적화되었다는 의미이므로


'''
batch_size
=6 >>> result 6.4142504
=5 >>> result 7.314257
=4 >>> result 7.086614
=3 >>> result 8.673927
=2 >>> result 6.241328
=1 >>> result 6.8972254
'''