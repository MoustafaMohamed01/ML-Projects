import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
!{sys.executable} -m pip install quandl

import quandl

df = pd.read_csv('NSE-TATAGLOBAL11.csv')
print(df.head())

import datacmp
print(datacmp.get_detailed(df))

plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Closing Price')
plt.show()

df['Open - Close'] = df['Open'] - df['Close']
df['High - Low'] = df['High'] - df['Low']
df = df.dropna()

X = df[['Open - Close', 'High - Low']]
X.head()

Y = np.where(df['Close'].shift(-1) > df['Close'],1,-1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=44)

from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

params = {'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn = neighbors.KNeighborsClassifier()
model = GridSearchCV(knn, params, cv=5)

model.fit(X_train, y_train)

accuracy_train = accuracy_score(y_train, model.predict(X_train))
accuracy_test = accuracy_score(y_test, model.predict(X_test))

print(f'Train Accuracy: {accuracy_train}')
print(f'Test Accuracy: {accuracy_test}')



