import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator: 이미지를 데이터로 변경 + 증폭해주는 기능
train_datagen = ImageDataGenerator(
    rescale=1./255, horizontal_flip=True, vertical_flip=True,
    width_shift_range=0.1, height_shift_range=0.1,
    rotation_range=5, zoom_range=1.2, shear_range=0.7,
    fill_mode='nearest')

# test 데이터에는 rescale만 하고 별다른 증폭을 하지 않는 이유?
# test 데이터의 목적은 실전과 유사한 평가이기 때문에
test_datagen = ImageDataGenerator(rescale=1./255)

# ImageDataGenerator로 생성한 객체를 Numpy Array로 변환해줌
# 폴더별로 y값 붙여 줌 (0, 1, 2, ...)
xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',  
    target_size=(200, 200),  # 모든 이미지를 압축 또는 증폭해 동일한 사이즈로 맞춰줌
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    # Found 160 images belonging to 2 classes.
    # 2개의 클래스를 가진 총 160개의 이미지 있음
)
# 이미지 픽셀 개수가 150*150, 이미지 장수가 총 160장이므로
# x.shape (160, 150, 150, 1)
# y.shape (160,)
# y.unique [0, 1]

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',  
    target_size=(200, 200),  # 모든 이미지를 압축 또는 증폭해 동일한 사이즈로 맞춰줌
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    # Found 120 images belonging to 2 classes.
    # 2개의 클래스를 가진 총 120개의 이미지 있음
)

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x0000019FB3997430>

print(xy_train[0])  # batch_size만큼씩 끊은 이미지의 첫번째 덩어리 (여기서는 이미지 10장)
# 총 160장이고 10장씩 나눠 담았으므로 총 16개 덩어리 -> xy_train[15]까지 존재함

print(xy_train[0][0].shape)  # x의 shape  # (10, 200, 200, 1)
# (batch_size, target_size, target_size, color??)
print(xy_train[0][1].shape)  # y의 shape  # (10,)

print(type(xy_train))  # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))  # <class 'tuple'>
print(type(xy_train[0][0]))  # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))  # <class 'numpy.ndarray'>
# batch size만큼씩 자른 numpy array 형태의 x, y를
# tuple에 담아서
# tensorflow keras 데이터 형태로 모아둠
# ex) xy_train[0][1][2] -> 첫번째 batch의 y 묶음 중 3번째 값