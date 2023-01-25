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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(2, 2), input_shape=(28, 28, 1),
                 padding='same', activation='relu'))  # (28, 28, 128)
model.add(MaxPooling2D())                             # (14, 14, 128)
model.add(Conv2D(filters=128, kernel_size=(2, 2),
                 padding='same', activation='relu'))  # ()
model.add(Conv2D(filters=64, kernel_size=(2, 2),
                 padding='same', activation='relu'))  # ()
model.add(Flatten())
model.add(Dense(100, activation='relu'))  # input_shape = (60000, 40000)
                                        #      (batch_size, input_dim)
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))


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



'''
model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(2, 2), input_shape=(28, 28, 1),
                 padding='same', activation='relu'))  # (28, 28, 128)
model.add(MaxPooling2D())                             # (14, 14, 128)
model.add(Conv2D(filters=128, kernel_size=(2, 2),
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
loss: 0.04240766167640686
acc: 0.989300012588501
'''