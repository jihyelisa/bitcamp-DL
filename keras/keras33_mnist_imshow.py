import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 이미 train/test 분리가 되어 있는 데이터이므로 나눌 필요 없음

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
                                     # 뒤 1이 생략됨 - 흑백데이터
print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])  # 5

import matplotlib.pyplot as plt
plt.imshow(x_train[927], 'gray')
plt.show()