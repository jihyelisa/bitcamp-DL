import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score,accuracy_score


#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split (
    x, y, shuffle=True,
    random_state=333, test_size=0.2
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model =  Sequential()
model.add(Dense(150, input_dim=(30)))
model.add(Dense(120, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

              
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy']
              )
                
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', 
                              patience=5, restore_best_weights=True
                              )

hist = model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2, verbose=3) 

loss, accuracy = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

y_predict = list(map(int, y_predict))
acc = accuracy_score(y_test, y_predict)

print('============================================')
print(hist)
print('============================================')
print(hist.history)
print('============================================')
print(hist.history['loss'])
print('============================================')
print(hist.history['val_loss'])
print('============================================')
print('loss : ', loss, ' accuracy : ', accuracy )
print('============================================')
print(' accuracy_score : ', acc )
print('============================================')


'''
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue',  marker='.' , label='val_loss' )
plt.grid() 
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('boston loss')
plt.legend()
plt.show()
'''
