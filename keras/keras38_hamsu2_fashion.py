from tensorflow.keras.datasets import fashion_mnist
import numpy as np


##1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# 이미 train/test 분리가 되어 있는 데이터이므로 나눌 필요 없음
'''
print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
                                     # 뒤 1이 생략됨 - 흑백데이터
print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)
'''

# 인풋 데이터의 shape를 3차원으로 맞춰주어야 함
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


##2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Input, Flatten, Dropout, MaxPooling2D

input1 = Input(shape=(28, 28, 1))
conv1 = Conv2D(filters=160, kernel_size=(4, 4), padding='same',
               strides=2, activation='relu')(input1)
maxpooling1 = MaxPooling2D()
conv2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
               strides=2, activation='relu')(conv1)
flatten = Flatten()(conv2)
dense1 = Dense(128, activation='relu')(flatten)
drop1 = Dropout(0.25)(dense1)
dense2 = Dense(64, activation='relu')(drop1)
output1 = Dense(10, activation='softmax')(dense2)
model = Model(inputs=input1, outputs=output1)


##3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

es = EarlyStopping(monitor="val_loss", mode="min",
                    patience=15, restore_best_weights=True)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=3, save_best_only=True,
                      filepath=filepath + 'fashion_' + date + '_' + filename)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, verbose=3,
          validation_split=0.1, callbacks=[es, mcp], batch_size=100)


##4. 평가, 예측

results = model.evaluate(x_test, y_test)
print('loss:', results[0])
print('acc:', results[1])



'''
model = Sequential()
model.add(Conv2D(filters=200, kernel_size=(4, 4), input_shape=(28, 28, 1),
                 padding='same', activation='relu'))  # (28, 28, 128)
model.add(MaxPooling2D())                             # (14, 14, 128)
model.add(Conv2D(filters=128, kernel_size=(3, 3),
                 padding='same', activation='relu'))  # ()
model.add(Conv2D(filters=64, kernel_size=(2, 2),
                 padding='same', activation='relu'))  # ()
model.add(Flatten())
model.add(Dense(100, activation='relu'))  # input_shape = (60000, 40000)
                                        #      (batch_size, input_dim)
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

결과
loss: 0.2755594849586487
acc: 0.9093000292778015
'''