import numpy as np

x_train = np.load('./_data/brain/brain_x_train.npy')
y_train = np.load('./_data/brain/brain_y_train.npy')
x_test = np.load('./_data/brain/brain_x_test.npy')
y_test = np.load('./_data/brain/brain_y_test.npy')

# print(x_train.shape, x_test.shape)  # (160, 200, 200, 1) (120, 200, 200, 1)
# print(y_train.shape, y_test.shape)  # (160,) (120,)



##2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(120, (2,2), input_shape=(200, 200, 1)))
model.add(Conv2D(80, (3,3), activation='relu'))
model.add(Conv2D(40, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



##3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['acc'])

hist = model.fit(x_train, y_train,
                 batch_size=16, epochs=100,
                 validation_data=(x_test, y_test), verbose=3)



##4. 평가, 예측
loss = hist.history['loss']
val_loss = hist.history['val_loss']
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('accuracy:', accuracy[-1])
print('val_acc:', val_acc[-1])



'''
loss: 3.6270205328037264e-06
val_loss: 0.023630639538168907
accuracy: 1.0
val_acc: 0.9916666746139526
'''