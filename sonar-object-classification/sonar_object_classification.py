import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('data/sonar.all-data', header=None)
df.columns = [f'feature_{i}' for i in range(60)] + ['label']

import datacmp
print(datacmp.get_detailed(df))


X = df.drop(columns='label', axis=1)
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, stratify=y, random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)

X_train_pred = model.predict(X_train)
training_data_acc = accuracy_score(y_train, X_train_pred)
print("Training Accuracy:", training_data_acc)

X_test_pred = model.predict(X_test)
testing_data_acc = accuracy_score(y_test, X_test_pred)
print("Testing Accuracy:", testing_data_acc)

print("Classification Report:\n", classification_report(y_test, X_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, X_test_pred))

input_data = (
    0.0453, 0.0523, 0.0843, 0.0689, 0.1183, 0.2583, 0.2156, 0.3481, 0.3337, 0.2872,
    0.4918, 0.6552, 0.6919, 0.7797, 0.7464, 0.9444, 1.0000, 0.8874, 0.8024, 0.7818,
    0.5212, 0.4052, 0.3957, 0.3914, 0.3250, 0.3200, 0.3271, 0.2767, 0.4423, 0.2028,
    0.3788, 0.2947, 0.1984, 0.2341, 0.1306, 0.4182, 0.3835, 0.1057, 0.1840, 0.1970,
    0.1674, 0.0583, 0.1401, 0.1628, 0.0621, 0.0203, 0.0530, 0.0742, 0.0409, 0.0061,
    0.0125, 0.0084, 0.0089, 0.0048, 0.0094, 0.0191, 0.0140, 0.0049, 0.0052, 0.0044
)

input_data_as_numpy = np.asarray(input_data).reshape(1, -1)
input_data_scaled = scaler.transform(input_data_as_numpy)

prediction = model.predict(input_data_scaled)
print(f"The object is a {'Rock' if prediction[0] == 'R' else 'Mine'}")
