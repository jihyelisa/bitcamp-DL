from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()

# 인풋은 (60000=N, 5, 5, 1)
# (데이터 수, 가로 길이, 세로 길이, 채널(컬러/필터))
# (batch_size, rows, column, channels)
model.add(Conv2D(filters=10, kernel_size=(2, 2),
                 input_shape=(5, 5, 1)))            # (N, 4, 4, 10)
                 # 5x5로 쪼갠 흑백 이미지를 넣고,
                 # 2x2마다 자른 후 이어붙여,
                 # 10장으로 늘리겠다는 의미
                 # filter - output과 유사한 개념
model.add(Conv2D(filters=5, kernel_size=(2, 2)))    # (N, 3, 3, 5)
model.add(Flatten())  # 이미지를 일렬로 쭉 펴줌       # (N, 45)
model.add(Dense(10))                                # (N, 10)
            # 인풋은 (batch_size, input_dim)
model.add(Dense(1))                                 # (N, 1)

model.summary()
'''
Model: "sequential"
_________________________________________________________________        
 Layer (type)                Output Shape              Param #
=================================================================        
 conv2d (Conv2D)             (None, 4, 4, 10)          50

 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205

 flatten (Flatten)           (None, 45)                0

 dense (Dense)               (None, 10)                460

 dense_1 (Dense)             (None, 1)                 11

=================================================================        
Total params: 726
Trainable params: 726
Non-trainable params: 0
_________________________________________________________________ 

None은 이미지 데이터 장수를 의미
'''