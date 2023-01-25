import numpy as np
from tensorflow.keras.datasets import mnist


##1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

'''
print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)
'''

# 인풋 데이터의 shape를 3차원으로 맞춰주어야 함
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

'''
print(x_train.shape, y_train.shape)  # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28, 1) (10000,)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))
'''


##2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Input, Flatten, Dropout

input1 = Input(shape=(28, 28, 1))
conv1 = Conv2D(filters=128, kernel_size=(2, 2), activation='relu')(input1)
conv2 = Conv2D(filters=64, kernel_size=(2, 2), activation='relu')(conv1)
conv3 = Conv2D(filters=64, kernel_size=(2, 2), activation='relu')(conv2)
flatten = Flatten()(conv3)
dense1 = Dense(100, activation='relu')(flatten)
drop1 = Dropout(0.25)(dense1)
dense2 = Dense(50, activation='relu')(drop1)
output1 = Dense(10, activation='softmax')(dense2)
model = Model(inputs=input1, outputs=output1)


##3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

es = EarlyStopping(monitor="val_loss", mode="min",
                    patience=20, restore_best_weights=True)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=1, save_best_only=True,
                      filepath=filepath + 'mnist_' + date + '_' + filename)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, verbose=3,
          validation_split=0.1, callbacks=[es, mcp], batch_size=50)


##4. 평가, 예측

results = model.evaluate(x_test, y_test)
print('loss:', results[0])
print('acc:', results[1])



# [과제] earlystopping, modelcheckpoint, val 적용

'''
input1 = Input(shape=(28, 28, 1))
conv1 = Conv2D(filters=128, kernel_size=(2, 2), activation='relu')(input1)
conv2 = Conv2D(filters=64, kernel_size=(2, 2), activation='relu')(conv1)
conv3 = Conv2D(filters=64, kernel_size=(2, 2), activation='relu')(conv2)
flatten = Flatten()(conv3)
dense1 = Dense(100, activation='relu')(flatten)
dropout = Dropout(0.25)(dense1)
dense2 = Dense(50, activation='relu')(dropout)
output1 = Dense(10, activation='softmax')(dense2)
model = Model(inputs=input1, outputs=output1)

결과
loss: 0.05813582241535187
acc: 0.983299970626831
'''