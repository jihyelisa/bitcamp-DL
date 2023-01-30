import numpy as np


##1. 데이터

x_datasets = np.array([range(100), range(301, 401)]).transpose()
# print(x1_datasets.shape)  # (100, 2)

y1 = np.array(range(2001, 2101))  #(100,)
y2 = np.array(range(201, 301))  #(100,)

from sklearn.model_selection import train_test_split
# 3개 이상의 데이터셋도 train_test_split로 분리 가능
x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        x_datasets, y1, y2, train_size=0.85, random_state=123)

# print(x1_train.shape, x2_train.shape, x3_train.shape, y_train.shape)
# (70, 2) (70, 3) (70, 2) (70,)
# print(x1_test.shape, x2_test.shape, x3_test.shape, y_test.shape)
# (30, 2) (30, 3) (30, 2) (30,)



##2. 모델구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

###2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(14, activation='relu', name='dns11')(input1)
dense2 = Dense(20, activation='relu', name='dns12')(dense1)
dense3 = Dense(13, activation='relu', name='dns13')(dense2)
output1 = Dense(14, activation='relu', name='dns14')(dense3)

# ###2-2. 모델2
# input2 = Input(shape=(3,))
# dense21 = Dense(21, activation='relu', name='dns21')(input2)
# dense22 = Dense(22, activation='relu', name='dns22')(dense21)
# output2 = Dense(23, activation='relu', name='dns23')(dense22)

# ###2-3. 모델3
# input3 = Input(shape=(2,))
# dense31 = Dense(11, activation='relu', name='dns31')(input3)
# dense32 = Dense(24, activation='relu', name='dns32')(dense31)
# output3 = Dense(12, activation='relu', name='dns33')(dense32)

# ###2-4. 모델병합
from tensorflow.keras.layers import concatenate, Concatenate  # #3에 이용하기
# merge1 = concatenate([output1, output2, output3], name='mrg1')
# merge2 = Dense(12, activation='relu', name='mrg2')(merge1)
# merge3 = Dense(13, activation='relu', name='mrg3')(merge2)
# merge_output = Dense(10, name='mrg_out')(merge3)

###2-5. 모델분기
dense41 = Dense(12, activation='relu', name='dns41')(output1)
dense42 = Dense(24, activation='relu', name='dns42')(dense41)
output4 = Dense(2, activation='relu', name='dns43')(dense42)

dense51 = Dense(31, activation='relu', name='dns51')(output1)
dense52 = Dense(32, activation='relu', name='dns52')(dense51)
output5= Dense(2, activation='relu', name='dns53')(dense52)

###2-6. 모델선언
model = Model(inputs=input1, outputs=[output4, output5])



##3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, [y1_train, y2_train],
          epochs=100, batch_size=1, verbose=3)



##4. 평가, 예측

loss = model.evaluate(x_test, [y1_test, y2_test])
print('\nloss:', loss[0])

""""""

'''
loss: 30104.744140625
'''