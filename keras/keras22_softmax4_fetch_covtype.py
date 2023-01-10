import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


##1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape)  # (581012, 54) (581012,)

'''
tensorflow 이용하기

y = to_categorical(y)
print(x.shape, y.shape)  # (581012, 54) (581012, 8)

# print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
# 원본 데이터의 칼럼 array가 0이 아닌 1부터 시작
# 필요한 열이 실제로는 7개인데 8개로 늘어남

# 쉐이프 맞추는 작업 해야 함
y = np.delete(y, 0, axis=1)  # 0번째 칼럼을 지워줌
print(x.shape, y.shape)  # (581012, 54) (581012, 7)
'''



'''
sklearn 이용하기
'''

from sklearn.preprocessing import OneHotEncoder

# y를 2차원 데이터로 변환
y = y.reshape(-1,1)
print(x.shape, y.shape)  # (581012, 54) (581012, 1)

print(y[:10])
ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y)
y = y.toarray()
print(y[:10])


'''
pandas 이용하기

y = pd.get_dummies(y)
print(y.shape)  # (581012, 7)
print(y[:10])  # pandas는 컬럼을 series라는 타입으로 다룸
print(y.values, y.to_numpy())  # Numpy의 ndarray 형태로 바꿔줌
y = y.to_numpy()  # 두 가지 방법 중 .to_numpy()가 권장됨
'''


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123,
    test_size=0.2, stratify=y)



##2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(54,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(7, activation='softmax'))


##3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor="val_loss",
                              mode="min", patience=10, restore_best_weights=True)

history = model.fit(
    x_train, y_train, epochs=20, batch_size=500, validation_split=0.2,
    callbacks=[earlyStopping], verbose=0
    )


##4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('accuray:', accuracy)
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
# 가장 큰 자릿값만 뽑아냄 (원핫 인코딩 한 것을 다시 원상복귀 시켜줌)

acc = accuracy_score(y_test, y_predict) # 원값과 예측값을 비교한 accuracy값 리턴
print(acc)