import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


df = pd.read_csv('dataset.csv')
print(df.head())

print(df.shape)

print(df.info())

print(df.isnull().sum())

df = df.dropna()

print(df.isnull().sum())


df.replace({'Loan_Status':{'N':0, 'Y':1}}, inplace=True)

print(df['Dependents'].value_counts())

df = df.replace(to_replace='3+', value=4)


sns.countplot(data=df, x='Education', hue='Loan_Status')
plt.show()

sns.countplot(data=df, x='Married', hue='Loan_Status')
plt.show()


df.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0}
}, inplace=True)


X = df.drop(columns=['Loan_ID', 'Loan_Status'],axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=2)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, y_train)

X_train_prediction = classifier.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction, y_train)
print(training_accuracy)

X_test_prediction = classifier.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, y_test)
print(test_accuracy)

