import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import np_utils

train_features, test_features, train_labels, test_labels = np.load("../citation_dataset/train_features.npy"), np.load(
    "../citation_dataset/test_features.npy"), np.load("../citation_dataset/train_labels.npy"), np.load(
    "../citation_dataset/test_labels.npy")
print(train_features.shape)
x_train=[]
x_test=[]
for i in train_labels:
    x_train.append(int(i)-1)
for i in test_labels:
    x_test.append(int(i)-1)
train_labels = np_utils.to_categorical(x_train, num_classes=10)
test_labels = np_utils.to_categorical(x_test, num_classes=10)
model = Sequential()

model.add(Dense(128, activation='relu', input_dim=64))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', input_dim=64))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', input_dim=64))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_features, train_labels,
          epochs=200,
          batch_size=1000, verbose=2)
score = model.evaluate(test_features, test_labels, batch_size=128)
print(score)
