from tensorflow.keras.datasets import cifar100  # 컬러 데이터임
import numpy as np


##1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

'''
print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)    # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))  # output 100
'''


##2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Input, Flatten, Dropout, MaxPooling2D

input1 = Input(shape=(32, 32, 3))
conv1 = Conv2D(filters=180, kernel_size=(4, 4), padding='same')(input1)
maxpooling1 = MaxPooling2D()
conv2 = Conv2D(filters=75, kernel_size=(2, 2), activation='relu')(maxpooling1)
conv3 = Conv2D(filters=50, kernel_size=(2, 2), activation='relu')(conv2)
flatten = Flatten()(conv3)
dense1 = Dense(80, activation='relu')(flatten)
dense2 = Dense(50, activation='relu')(dense1)
output1 = Dense(10, activation='softmax')(dense2)
model = Model(inputs=input1, outputs=output1)



model.add(Conv2D(filters=180, kernel_size=(4,4),
                 padding='same', input_shape=(32,32,3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=160, kernel_size=(2,2), padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=140, kernel_size=(2,2), padding='same'))
model.add(Flatten())
model.add(Dense(140, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='softmax'))



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
                      filepath=filepath + 'cifar100_' + date + '_' + filename)


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
model.add(Conv2D(filters=180, kernel_size=(2,2),
                 input_shape=(32,32,3)))
model.add(Conv2D(filters=160, kernel_size=(2,2)))
model.add(Conv2D(filters=140, kernel_size=(2,2)))
model.add(Flatten())
model.add(Dense(140, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='softmax'))

결과
loss: 4.605221748352051
acc: 0.009999999776482582
'''