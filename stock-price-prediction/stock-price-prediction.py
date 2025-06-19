import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import quandl
except ImportError:
    !{sys.executable} -m pip install quandl

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    df = pd.read_csv('NSE-TATAGLOBAL11.csv')
except FileNotFoundError:
    print("Error: 'NSE-TATAGLOBAL11.csv' not found. Please ensure the file is in the same directory.")
    sys.exit()

df['Open - Close'] = df['Open'] - df['Close']
df['High - Low'] = df['High'] - df['Low']
df.dropna(inplace=True)

X = df[['Open - Close', 'High - Low']].iloc[:-1]
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)[:-1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=44)

print(f"\nTraining data shape: {X_train.shape}, Test data shape: {X_test.shape}")

models = {
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params': {'C': [0.01, 0.1, 1, 10]}
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': [5, 7, 9, 11]}
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(random_state=44),
        'params': {'max_depth': [3, 5, 7]}
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=44),
        'params': {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=44),
        'params': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
    },
    'SVC': {
        'model': SVC(random_state=44),
        'params': {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
    },
    'XGBClassifier': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=44),
        'params': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
    }
}

best_model = None
best_accuracy = 0.0
best_model_name = ""
model_results = {}

for name, config in models.items():
    print(f"\nTraining {name}...")
    grid_search = GridSearchCV(config['model'], config['params'], cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, grid_search.predict(X_train))
    test_accuracy = accuracy_score(y_test, grid_search.predict(X_test))

    model_results[name] = {
        'best_estimator': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }

    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = grid_search.best_estimator_
        best_model_name = name

summary_data = []

for name, result in model_results.items():
    summary_data.append({
        'Model': name,
        'Best Parameters': result['best_params'],
        'Train Accuracy': round(result['train_accuracy'], 4),
        'Test Accuracy': round(result['test_accuracy'], 4)
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values(by='Test Accuracy', ascending=False).reset_index(drop=True)

print("\n--- 📊 Model Comparison Summary ---")
print(summary_df.to_markdown(index=False))


print(f"\n--- Best Model Identified ---")
if best_model:
    print(f"The best model is: {best_model_name}")
    print(f"With Test Accuracy: {best_accuracy:.4f}")

    model_filename = 'best_stock_prediction_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Best model saved as: {model_filename}")
else:
    print("No best model found. Something might have gone wrong with training.")



plt.style.use('dark_background')
sorted_results = dict(sorted(
    {name: res['test_accuracy'] for name, res in model_results.items()}.items(),
    key=lambda item: item[1], reverse=True
))
model_names = list(sorted_results.keys())
accuracies = list(sorted_results.values())

plt.figure(figsize=(12, 7))
palette = sns.color_palette("Spectral", len(model_names))
bars = sns.barplot(x=model_names, y=accuracies, palette=palette)

for i, bar in enumerate(bars.patches):
    height = bar.get_height()
    bars.annotate(f'{height:.2%}', (bar.get_x() + bar.get_width() / 2, height),
                  ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('Comparison of ML Model Test Accuracies', fontsize=16, fontweight='bold')
plt.xlabel('Model', fontsize=13)
plt.ylabel('Test Accuracy', fontsize=13)
plt.ylim(0, 1.05)
plt.xticks(rotation=45, fontsize=11)
plt.tight_layout()
plt.savefig("images/comparison-of-ml-models-acc.png", dpi=300, bbox_inches='tight')
plt.show()
