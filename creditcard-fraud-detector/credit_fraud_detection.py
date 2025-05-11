import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('creditcard.csv')
print(df.head())

import datacmp
print(datacmp.get_detailed(df))

print(df['Class'].value_counts())
print(df.groupby('Class').mean())

# Correlation Heatmap
plt.style.use('dark_background')
plt.figure(figsize=(25,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='viridis', linewidths=0.5, linecolor='black')
plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


cor_target = abs(corr['Class']).sort_values(ascending=False)[1:11]
plt.figure(figsize=(8, 5))
sns.barplot(x=cor_target.values, y=cor_target.index, palette='mako')
plt.title('Top 10 Features Most Correlated with Fraud (Class)', fontsize=14, fontweight='bold')
plt.xlabel('Correlation', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("plots/top10_correlated_features.png", dpi=300, bbox_inches='tight')
plt.show()

X = df.drop(columns='Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_pred, y_test)}')

import joblib
joblib.dump(model, 'model/xgboost_fraud_model.pkl')

