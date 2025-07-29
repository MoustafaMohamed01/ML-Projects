import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('dataset.csv')

print(df.head())

print('\n', df.shape)

print('\n', df.info())

print('\n', df.isnull().sum())

df = df.dropna()
print('\n', df.isnull().sum())

df = df.drop(columns=['Loan_ID'])

print(df.columns)


df['Dependents'] = df['Dependents'].replace('3+', 4).astype(int)

binary_cols = ['Gender', 'Married', 'Self_Employed', 'Loan_Status']

le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, columns=['Education', 'Property_Area'], drop_first=True)


plt.style.use('dark_background')
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Loan_Status', edgecolor='white', linewidth=2)
plt.title('Loan Status Distribution', fontsize=16, weight='bold')
plt.xlabel('Loan Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height() + 2), 
                ha='center', color='white', fontsize=10)
plt.tight_layout()
plt.savefig('graphs/loan_status_distribution.png', dpi=300, facecolor='black')
plt.show()


plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Married', hue='Loan_Status', edgecolor='white', linewidth=1.5)
plt.title('Loan Status by Marital Status', fontsize=16, weight='bold')
plt.xlabel('Marital Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Loan Status', loc='upper right')
plt.tight_layout()
plt.savefig('graphs/loan_status_by_married.png', dpi=300, facecolor='black')
plt.show()


def decode_property_area(row):
    if row['Property_Area_Semiurban'] == 1:
        return 'Semiurban'
    elif row['Property_Area_Urban'] == 1:
        return 'Urban'
    else:
        return 'Rural'

df_new = df.copy()
df_new['Property_Area'] = df_new.apply(decode_property_area, axis=1)

plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df_new, x='Property_Area', hue='Loan_Status', edgecolor='white', linewidth=1.5)
plt.title('Loan Status by Property Area', fontsize=16, weight='bold')
plt.xlabel('Property Area', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Loan Status', loc='upper right')
plt.tight_layout()
plt.savefig('graphs/loan_status_by_property_area.png', dpi=300, facecolor='black')
plt.show()


plt.figure(figsize=(8, 5))
ax = sns.histplot(data=df, x='ApplicantIncome', kde=True, color='#5ab4ac')
plt.title('Applicant Income Distribution', fontsize=16, weight='bold')
plt.xlabel('Applicant Income', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('graphs/applicant_income_distribution.png', dpi=300, facecolor='black')
plt.show()


plt.figure(figsize=(15, 10))
corr = df.corr(numeric_only=True)
ax = sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='white', square=True)
plt.title('Feature Correlation Heatmap', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('graphs/feature_correlation_heatmap.png', dpi=300, facecolor='black')
plt.show()


X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)


X_train_prediction = classifier.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction, y_train)
print('Training acc = ', training_accuracy)


X_test_prediction = classifier.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, y_test)
print('Test acc = ', test_accuracy)
