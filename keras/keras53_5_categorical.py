import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, horizontal_flip=True, vertical_flip=True,
    width_shift_range=0.1, height_shift_range=0.1,
    rotation_range=5, zoom_range=1.2, shear_range=0.7,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',  
    target_size=(200, 200),
    batch_size=10,
    class_mode='categorical',  # class_mode를 categorical로 지정하면 원핫인코딩 됨
    color_mode='grayscale',
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',  
    target_size=(200, 200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)


# class_mode='categorical'
print(xy_train[0][1])
'''
[[1. 0.]
 [0. 1.]
 [0. 1.]
 [1. 0.]
 [1. 0.]
 [1. 0.]
 [0. 1.]
 [0. 1.]
 [0. 1.]
 [0. 1.]]
 '''
 
# class_mode='binary'
print(xy_test[0][1])
'''
[1. 1. 1. 1. 1. 0. 1. 0. 1. 1.]
'''