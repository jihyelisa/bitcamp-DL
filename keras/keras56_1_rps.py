# 실습
# 가위바위보 모델 만들기


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
'''
datagen_augument = ImageDataGenerator(
    rescale=1./255, horizontal_flip=True, vertical_flip=True,
    width_shift_range=0.1, height_shift_range=0.1,
    rotation_range=5, zoom_range=1.2, shear_range=0.7,
    fill_mode='nearest')

xy_augument = datagen_augument.flow_from_directory(
    'D:/_data/rps/',
    target_size=(150, 150),
    batch_size=2520,  # 너무 큰 수를 주면 memory overflow 발생
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)  # Found 25000 images belonging to 2 classes.

datagen_origin = ImageDataGenerator(rescale=1./255)

xy_train = datagen_origin.flow_from_directory(
    'D:/_data/rps/',
    target_size=(150, 150),
    batch_size=2520,  # 너무 큰 수를 주면 memory overflow 발생
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

print(xy_augument[0][0].shape)  # (2520, 100, 100, 3)
print(xy_train[0][0].shape)  # (2520, 100, 100, 3)

x_train = np.concatenate((xy_train[0][0], xy_augument[0][0]))
y_train = np.concatenate((xy_train[0][1], xy_augument[0][1]))

np.save('./_data/rps/rps_x_train.npy', arr=x_train)
np.save('./_data/rps/rps_y_train.npy', arr=y_train)


'''
x_train = np.load('./_data/rps/rps_x_train.npy')
y_train = np.load('./_data/rps/rps_y_train.npy')

print(x_train.shape, y_train.shape)
# (5040, 100, 100, 3) (5040,)



##2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

model = Sequential()
model.add(Conv2D(30, 10, activation='relu', input_shape=(150, 150, 3)))
model.add(Conv2D(20, 5, activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Conv2D(10, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.summary()



##3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

hist = model.fit(x_train, y_train,
                 batch_size=160, epochs=5,
                 validation_split=0.3,
                 verbose=3)



##4. 평가, 예측
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']

# print('loss:', loss[-1])
# print('val_loss:', val_loss[-1])
print('accuracy:', accuracy[-1])
print('val_acc:', val_acc[-1])





import matplotlib.pyplot as plt

plt.figure(figsize = (12,8))

for each in ['loss', 'val_loss', 'acc', 'val_acc']:
    plt.plot(hist.history[each], label = each)
    
plt.legend()
plt.grid()
plt.show()



'''
accuracy: 0.3333333432674408
val_acc: 0.3333333432674408
'''