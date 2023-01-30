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
    batch_size=10,
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
model.add(Conv2D(64, (2,2), input_shape=(100, 100, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


##3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['acc'])

hist = model.fit_generator(xy_train,
                    # xy_train 안에
                    # 이미 x, y, batch_size가 지정되어 있음
                    steps_per_epoch=16,
                    # 총 160개의 훈련 샘플이 있고
                    # batch size가 10이므로
                    # steps_per_epoch는 최대 16
                    epochs=200,
                    validation_data=xy_test,
                    validation_steps=4,
                    # 한 epoch 종료 후 검증 시의 step 수
                    verbose=3)



# 과제
# matplomap으로 그림 그리기!!

##4. 오차, 차트

loss = hist.history['loss']
val_loss = hist.history['val_loss']
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('accuracy:', accuracy[-1])
print('val_acc:', val_acc[-1])


import matplotlib.pyplot as plt
import matplotlib

# 차트 한글 폰트 사용
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9,6))
# x 명시 안해도 됨
# hist loss 사용, 색은 red, 선모양은 ., y 선의 이름은 loss
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
# hist val_loss 사용, 색은 blue, 선모양은 ., x 선의 이름은 val_loss
plt.plot(hist.history['val_loss'], c='blue',  marker='.' , label='val_loss' )
# 차트 gird 생성
plt.grid() 
# x 축 이름
plt.xlabel('epochs')
# y 축 이름
plt.ylabel('loss')
# 차트 제목
plt.title('brain chart')
# 그래프 선 이름 표
plt.legend()
#plt.legend(loc='upper right')  그래프 선 이름 표, 위치
plt.show()



import matplotlib.pyplot as plt
plt.imshow(xy_test[10][0][0], 'gray')
plt.show()


'''
loss: 0.37785300612449646
val_loss: 0.23435744643211365
accuracy: 0.831250011920929
val_acc: 0.949999988079071
'''