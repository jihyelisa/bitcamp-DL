import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


##1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255, horizontal_flip=True, vertical_flip=True,
    width_shift_range=0.1, height_shift_range=0.1,
    rotation_range=5, zoom_range=1.2, shear_range=0.7,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',  
    target_size=(100, 100),
    batch_size=1000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',  
    target_size=(100, 100),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)



##2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(120, (2,2), input_shape=(100, 100, 1)))
model.add(Conv2D(80, (3,3), activation='relu'))
model.add(Conv2D(40, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



##3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['acc'])

# hist = model.fit_generator(xy_train,
#                     # xy_train 안에
#                     # 이미 x, y, batch_size가 지정되어 있음
#                     steps_per_epoch=16,
#                     # 총 160개의 훈련 샘플이 있고
#                     # batch size가 10이므로
#                     # steps_per_epoch는 최대 16
#                     epochs=200,
#                     validation_data=xy_test,
#                     validation_steps=4,
#                     # 한 epoch 종료 후 검증 시의 step 수
#                     verbose=3)

hist = model.fit(xy_train[0][0], xy_train[0][1],
                 # ImageDataGenerator에서 배치 사이즈를 데이터 양보다 크게 잡아줌
                 # xy_train[0][0]에는 모든 x가, xy_train[0][1]에는 모든 y가 들어 있음
                 batch_size=4, epochs=150, verbose=3,
                 validation_data=(xy_test[0][0], xy_test[0][1]))



##4. 평가, 예측
loss = hist.history['loss']
val_loss = hist.history['val_loss']
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('accuracy:', accuracy[-1])
print('val_acc:', val_acc[-1])



'''
loss: 0.6931840181350708
val_loss: 0.6931349039077759
accuracy: 0.44999998807907104
val_acc: 0.6000000238418579
'''