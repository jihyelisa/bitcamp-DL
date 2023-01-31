import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True, vertical_flip=True,
    # width_shift_range=0.1, height_shift_range=0.1,
    # rotation_range=5, zoom_range=1.2, shear_range=0.7,
    # fill_mode='nearest'
    )

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',  
    target_size=(200, 200),
    batch_size=10000,  # 한 묶음에 담기도록 batch_size를 데이터 양보다 크게 해줌
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',  
    target_size=(200, 200),
    batch_size=10000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)



# 실행 시마다 새로 수치화부터 하면 시간이 많이 걸림
# 이미지를 수치화한 numpy 파일로 저장해놓음
# 데이터로 주어지는 이미지에 따라 용량을 줄이는 효과도 있음

# x, y 한 번에 저장은 불가 > 자료형이 numpy array가 아닌 tuple이므로
# np.save('./_data/brain/brain_xy_train.npy', arr=xy_train[0])

# x, y 각각 저장
np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])
np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])
np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])