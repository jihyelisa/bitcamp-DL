import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


##1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True))
# y값이 label 별로 각각 몇 개인지
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(x.shape, y.shape)  # (178, 13) (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123,
    test_size=0.1, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



'''
##2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(13,)))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))
'''

##2. 함수형 모델구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
dense1 = Dense(100, activation='relu')(input1)
dense2 = Dense(80, activation ='relu')(dense1)
dense3 = Dense(60, activation ='relu')(dense2)
dense4 = Dense(40, activation ='relu')(dense3)
dense5 = Dense(20, activation ='relu')(dense4)
output1 = Dense(3, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1)



##3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=20,
          validation_split=0.1, verbose=3)


##4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)


from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict) # 원값과 예측값을 비교한 accuracy값 리턴
print("accuracy_score:", acc)


'''
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123,
    test_size=0.1, stratify=y)


##2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(13,)))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))


##3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=20,
          validation_split=0.1, verbose=3)



# scaler 전 결과
accuracy_score: 0.9444444444444444

# scaler 후 결과
accuracy_score: 1.0
'''