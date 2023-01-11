from sklearn.datasets import load_iris 
# 꽃잎 등을 보고 어떤 꽃인지 분석함
# 4개의 칼럼으로 어떤 클래스인지 알아냄

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

##1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets['target']
print(x.shape)  # (150, 4)
print(y.shape)  # (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123,
    test_size=0.2,
    stratify=y
)


##2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(4,)))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))


##3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=1,
          validation_split=0.2, verbose=3)


##4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('accuray:', accuracy)
print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)


from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
# y_test는 원핫인코딩이나 loss=sparse_categorical_crossentropy를 거치지 않았으므로 argmax로 원복할 필요 없음

acc = accuracy_score(y_test, y_predict)
print(acc)