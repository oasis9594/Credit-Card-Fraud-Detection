import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import SGD
import tensorflow as tf

import pandas_ml as pdml
import imblearn

df = pd.read_csv('creditcard.csv', low_memory=False)
X = df.iloc[:,:-1]
y = df['Class']

df.head()

from pandas_ml import ConfusionMatrix

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

data = scale(X)
pca = PCA(n_components=10)
X = pca.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

df = pdml.ModelFrame(X_train, target=y_train)
sampler = df.imbalance.over_sampling.SMOTE()
oversampled = df.fit_sample(sampler)
X2, y2 = oversampled.iloc[:,1:11], oversampled['Class']
print(X2)
print(y2)
X2=X2.as_matrix()
y2=y2.as_matrix()

model = Sequential()
model.add(Dense(27, input_dim=10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

h = model.fit(X2, y2, epochs=5, validation_data=(X_test, y_test))

print("Loss: ", model.evaluate(X_test, y_test, verbose=2))
y_predicted = np.round(model.predict(X_test)).T[0]
y_correct = np.array(y_test)

confusion_matrix = ConfusionMatrix(y_correct, y_predicted)
confusion_matrix.plot(normalized=True)
plt.show()
#confusion_matrix2.print_stats()

false_neg=0.0
false_pos=0.0
true_pos=0.0
true_neg=0.0
incorrect=0.0
total=len(y_predicted)

for i in range(len(y_predicted)):
	if y_predicted[i]!=y_correct[i] :
		incorrect+=1
		if y_predicted[i] == 1 and y_correct[i] == 0 :
			false_pos+=1
		else :
			false_neg+=1
	else :
		if y_predicted[i] == 1 and y_correct[i] == 1 :
			true_pos+=1
		else :
			true_neg+=1
print("TP: ", true_pos/total)
print("FP: ", false_pos/total)
print("TN: ", true_neg/total)
print("FN: ", false_neg/total)
inaccuracy = incorrect/total
accuracy = 1 - inaccuracy
print("Accuracy: ", accuracy)

recall = 0.0
recall = true_pos/(true_pos+false_neg)
precision = true_pos/(true_pos + false_pos )

print("Recall: ", recall)
print("Precision: ", precision)