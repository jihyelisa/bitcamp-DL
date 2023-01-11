from sklearn.datasets import load_iris

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

##1. 데이터
datasets = load_iris()

print(datasets.feature_names)   #  pandas에서는 .columns

x = datasets.data
y = datasets['target']
print(x.shape)  # (150, 4)
print(y.shape)  # (150,)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123,
    test_size=0.2,
    stratify=y 
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

'''
##2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(4,)))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))
'''

##2. 함수형 모델구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(4,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation ='sigmoid')(dense1)
dense3 = Dense(30, activation ='relu')(dense2)
dense4 = Dense(20, activation ='relu')(dense3)
output1 = Dense(3, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)


##3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=1,
          validation_split=0.2, verbose=0)


##4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)


from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict) # 원값과 예측값을 비교한 accuracy값 리턴

print('accuray:', accuracy)
print('accuracy_score:', acc)

'''
# scaler 전
accuray: 0.9666666388511658
accuracy_score: 0.9666666666666667

# scaler 후
accuray: 0.8333333134651184
accuracy_score: 0.8333333333333334
'''