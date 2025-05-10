import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('data/diabetes.csv')
print(df.head())

print(df.shape)
print(df.describe())
print(df['Outcome'].value_counts())
print(df.groupby('Outcome').mean())

print((df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]==0).sum())

invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[invalid_cols] = df[invalid_cols].replace(0, np.nan)
df[invalid_cols] = df[invalid_cols].fillna(df[invalid_cols].median())

# Distribution of Diabetes Outcomes
plt.style.use('dark_background')
plt.figure(figsize=(6, 4))

countplot = sns.countplot(x='Outcome', data=df, palette='Set2')
countplot.set_title('Distribution of Diabetes Outcomes', fontsize=16, color='white')
countplot.set_xlabel('Diabetes (0 = No, 1 = Yes)', fontsize=12, color='white')
countplot.set_ylabel('Count', fontsize=12, color='white')

plt.savefig('images/diabetes_countplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Pairplot of Diabetes Features
plt.style.use('dark_background')

cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age']

plot = sns.pairplot(df[cols + ['Outcome']], hue='Outcome', palette='coolwarm', diag_kind='kde')
plot.fig.suptitle('Pairplot of Diabetes Features', fontsize=16, color='white')
plot.fig.tight_layout()
plot.fig.subplots_adjust(top=0.95)

plot.savefig("images/diabetes_pairplot.png", dpi=300, bbox_inches='tight')
plt.show()

# Feature Correlation Heatmap
plt.style.use('dark_background')

plt.figure(figsize=(15, 6))

heatmap = sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm',
          annot_kws={"color": "white"}, linewidths=0.5, linecolor='black')
plt.title('Feature Correlation Heatmap', fontsize=16, color='white')

plt.savefig("images/diabetes_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()


X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "SVM (Linear)": svm.SVC(kernel='linear'),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")

model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy :", accuracy_score(y_test, test_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_pred))
print("Classification Report:", classification_report(y_test, test_pred))

input_data = (1, 189, 40, 27, 846, 30.1, 0.398, 59)
input_np = np.asarray(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_np)
prediction = model.predict(input_scaled)[0]

if prediction == 0:
    print('The person does NOT have diabetes.')
else:
    print('The person HAS diabetes.')

