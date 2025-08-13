import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('winequality-red.csv')

print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

print(df.columns)

plt.figure(dpi=300)
sns.countplot(x='quality', data=df)
plt.show()

plot = plt.figure(figsize=(5, 5), dpi=300)
sns.barplot(x='quality', y='volatile acidity', data=df)
plt.show()

plot = plt.figure(figsize=(5, 5), dpi=300)
sns.barplot(x='quality', y='citric acid', data=df)
plt.show()

corr = df.corr()
plt.figure(figsize=(5, 5), dpi=300)
sns.heatmap(corr, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()

X = df.drop('quality', axis=1)
y = df['quality'].apply(lambda y_value:1 if y_value>=7 else 0)

print(y.unique())
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

X_test_prediction = model.predict(X_test)

test_data_acc = accuracy_score(X_test_prediction, y_test)

print(test_data_acc)

input_data = (7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if prediction[0] == 1:
    print('Good Quality')
else:
    print('Bad Quality')

