from tensorflow.keras.datasets import cifar10  # 컬러 데이터임
import numpy as np


##1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

'''
print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000,)
print(x_test.shape, y_test.shape)    # (10000, 32, 32, 3) (10000,)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
# array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
# dtype=int64))
'''


##2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=150, kernel_size=(2, 2),
                 input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(filters=100, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(filters=50, kernel_size=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))



##3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

es = EarlyStopping(monitor="val_loss", mode="min",
                    patience=10, restore_best_weights=True)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=1, save_best_only=True,
                      filepath=filepath + 'cifar10_' + date + '_' + filename)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, verbose=3,
          validation_split=0.125, callbacks=[es, mcp], batch_size=100)


##4. 평가, 예측

results = model.evaluate(x_test, y_test)
print('loss:', results[0])
print('acc:', results[1])
