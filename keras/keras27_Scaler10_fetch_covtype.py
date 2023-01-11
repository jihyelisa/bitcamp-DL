import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


##1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

from tensorflow.keras.utils import to_categorical

y = to_categorical(y)
y = np.delete(y, 0, axis=1)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123,
    test_size=0.1, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


##2. 모델구성
model = Sequential()
model.add(Dense(300, activation='relu', input_shape=(54,)))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(7, activation='softmax'))


##3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor="val_loss",
                              mode="min", patience=5, restore_best_weights=True)

history = model.fit(
    x_train, y_train, epochs=50, batch_size=200, validation_split=0.125,
    callbacks=[earlyStopping], verbose=3
    )


##4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('accuray:', accuracy)
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict) # 원값과 예측값을 비교한 accuracy값 리턴
print(acc)



'''
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123,
    test_size=0.1, stratify=y)


##2. 모델구성
model = Sequential()
model.add(Dense(300, activation='relu', input_shape=(54,)))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(7, activation='softmax'))


##3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor="val_loss",
                              mode="min", patience=10, restore_best_weights=True)

history = model.fit(
    x_train, y_train, epochs=100, batch_size=200, validation_split=0.125,
    callbacks=[earlyStopping], verbose=3
    )



# scaler 사용 전
accuray: 0.9342018961906433
0.9342019207600427

# scaler 사용 후
accuray: 0.9506901502609253
0.9506901655708926
'''