import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    'D:/_data/dogs-vs-cats/train/',
    classes=['cat','dog'],
    # cat, dog directory 순서로 label 0,1을 설정
    # 선생님은 생략하셨음
    target_size=(100, 100),
    batch_size=26000,  # 너무 큰 수를 주면 memory overflow 발생
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
)  # Found 25000 images belonging to 2 classes.

# xy_test = test_datagen.flow_from_directory(
#     'D:/_data/dogs-vs-cats/test/',
#     target_size=(100, 100),
#     batch_size=13000,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=True,
# )

test_img = test_datagen.flow_from_directory(
    'D:/_data/dogs-vs-cats/test_img/',
    target_size=(100, 100),
    batch_size=6,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
)

# np.save('./_data/dogs-vs-cats/dog_cat_x_train.npy', arr=xy_train[0][0])
# np.save('./_data/dogs-vs-cats/dog_cat_y_train.npy', arr=xy_train[0][1])
# np.save('./_data/dogs-vs-cats/dog_cat_x_test_100.npy', arr=xy_test[0][0])
# np.save('./_data/dogs-vs-cats/dog_cat_y_test_100.npy', arr=xy_test[0][1])
np.save('./_data/dogs-vs-cats/test_img.npy', arr=test_img[0][0])