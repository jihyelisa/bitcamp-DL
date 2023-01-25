from tensorflow.keras.datasets import cifar10  # 컬러 데이터임
import numpy as np


##1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


##2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Input, Flatten, Dropout, MaxPooling2D

input1 = Input(shape=(32, 32, 3))
conv1 = Conv2D(filters=100, kernel_size=(2, 2), padding='same',
               strides=2, activation='relu')(input1)
maxpooling1 = MaxPooling2D()(conv1)
conv2 = Conv2D(filters=75, kernel_size=(2, 2), activation='relu')(maxpooling1)
conv3 = Conv2D(filters=50, kernel_size=(2, 2), activation='relu')(conv2)
flatten = Flatten()(conv3)
dense1 = Dense(80, activation='relu')(flatten)
dense2 = Dense(50, activation='relu')(dense1)
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
                      verbose=1, save_best_only=True,
                      filepath=filepath + 'cifar10_' + date + '_' + filename)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, verbose=3,
          validation_split=0.1, callbacks=[es, mcp], batch_size=100)


##4. 평가, 예측

results = model.evaluate(x_test, y_test)
print('loss:', results[0])
print('acc:', results[1])



'''
model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2),
                 input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=75, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=50, kernel_size=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(80, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

결과
loss: 0.9960728287696838
acc: 0.6672000288963318
'''