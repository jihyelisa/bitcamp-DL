import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


##1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape)
y = to_categorical(y)   # 원핫 인코딩
print(x.shape, y.shape) # (1797, 64) (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123,
    test_size=0.2, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


'''
##2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(64,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))
'''

##2. 함수형 모델구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(64,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation ='relu')(dense1)
dense3 = Dense(30, activation ='relu')(dense2)
dense4 = Dense(20, activation ='relu')(dense3)
output1 = Dense(10, activation ='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)


##3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True)

history = model.fit(
    x_train, y_train, epochs=200, batch_size=1, validation_split=0.2,
    callbacks=[earlyStopping], verbose=3
    )


##4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('accuray:', accuracy)
y_predict = model.predict(x_test[:5])


'''
print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y)) # [0 1 2 3 4 5 6 7 8 9]
print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))


import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[7])
plt.show()
'''


'''
# scaler 사용 전
accuray: 0.9722222089767456

# scaler 사용 후
accuray: 0.9638888835906982
'''