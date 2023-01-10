from sklearn.datasets import load_iris 
# 꽃잎 등을 보고 어떤 꽃인지 분석함
# 4개의 칼럼으로 어떤 클래스인지 알아냄

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

##1. 데이터
datasets = load_iris()
# print(datasets.DESCR)   # pandas의 .describe() / .info()
#  :Attribute Information:  # 꽃에 대한 정보 4가지
#         - sepal length in cm
#         - sepal width in cm
#         - petal length in cm
#         - petal width in cm
#         - class:  # 꽃의 종류 3가지
#                 - Iris-Setosa
#                 - Iris-Versicolour
#                 - Iris-Virginica

print(datasets.feature_names)   #  판다스 .columns

x = datasets.data
y = datasets['target']
print(x.shape)  # (150, 4)
print(y.shape)  # (150,)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)



x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123,
    # 분류문제에서 그냥 shuffle을 사용할 때 발생할 수 있는 문제
    # 데이터 상의 비율이 한쪽으로 치우쳐 있을 때,
    # 인공지능의 예측 결과도 높은 비율의 결과 쪽으로 치우칠 수 있다
    # ex. 암환자인 사람 10%, 암환자가 아닌 사람 90%
    #     대부분을 암환자가 아닌 사람으로 예측해도 제법 정확한 결과임
    test_size=0.2,
    stratify=y  # 결과 비율을 원래 데이터와 맞춰줌
)


##2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(4,)))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))
# 다중분류 activation: softmax

# softmax의 원리
# 결과의 선택지(여기서는 꽃의 종류) 별 비율을 합해 100%(=1)에 가깝게 나오도록 함
# 마지막 노드의 개수는 최종 선택지의 가짓수


##3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=1,
          validation_split=0.2, verbose=0)


##4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('accuray:', accuracy)
print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)


# 결과출력
# accuray: 0.9333333373069763
# 
# 좌표형태로 하나를 선택함
# [[0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# 
# 세 가지 predict 값 중 가장 높은 것이 선택됨
# [[1.3271718e-04 9.9982339e-01 4.3875436e-05]
#  [3.8900490e-07 1.2987216e-02 9.8701233e-01]
#  [9.9998355e-01 1.6466707e-05 1.9518398e-15]
#  [9.9999142e-01 8.5902184e-06 6.4293697e-16]
#  [8.1163678e-05 8.8541102e-01 1.1450777e-01]]


from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
# 가장 큰 자릿값만 뽑아냄 (원핫 인코딩 한 것을 다시 원상복귀 시켜줌)

acc = accuracy_score(y_test, y_predict) # 원값과 예측값을 비교한 accuracy값 리턴
print(acc)