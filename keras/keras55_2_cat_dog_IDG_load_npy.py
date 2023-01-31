import numpy as np

x_train = np.load('./_data/dogs-vs-cats/dog_cat_x_train_100.npy')
y_train = np.load('./_data/dogs-vs-cats/dog_cat_y_train_100.npy')
# x_test = np.load('./_data/dogs-vs-cats/dog_cat_x_test_100.npy')
# y_test = np.load('./_data/dogs-vs-cats/dog_cat_y_test_100.npy')
test_img = np.load('./_data/dogs-vs-cats/test_img.npy')

# print(x_train.shape, y_train.shape)
# (25000, 100, 100, 3) (25000,)
# print(x_test.shape, y_test.shape)
# (12500, 100, 100, 3) (12500,)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test \
    = train_test_split(x_train, y_train,
                       test_size = 0.15, random_state = 13,
                       stratify = y_train)



##2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(30, 3, input_shape=(100, 100, 3)))
model.add(Conv2D(30, 3, activation='relu'))
model.add(Conv2D(20, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))



##3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

hist = model.fit(x_train, y_train,
                 batch_size=290, epochs=4,
                 validation_data=[x_test, y_test],
                 verbose=3)



##4. 평가, 예측
loss = hist.history['loss']
val_loss = hist.history['val_loss']
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('accuracy:', accuracy[-1])
print('val_acc:', val_acc[-1])


test = model.predict(test_img)
test = np.argmax(test, axis=1)
print(test)

result = []
for i in range(5):
    if test[i] == 0:
        result.append('cat')
    else:
        result.append('dog')
print(f'We have {result}')
print(f'\nAnd Ddiyong is a good {result[0]}.')





import matplotlib.pyplot as plt
import matplotlib

plt.figure(figsize = (12,8))

for each in ['loss', 'val_loss', 'acc', 'val_acc']:
    plt.plot(hist.history[each], label = each)
    
plt.legend()
plt.grid()
plt.show()


'''

'''