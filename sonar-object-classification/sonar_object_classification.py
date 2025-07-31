import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

df = pd.read_csv('data/sonar data.csv', header=None)

print(df.info())

df.columns = [f'feature_{i}' for i in range(60)] + ['label']


plt.style.use('dark_background')
neon_colors = ['#08F7FE', '#FE53BB', '#F5D300', '#00ff41']

plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df, palette=neon_colors)
plt.title("Class Distribution", fontsize=14, color='#F5D300')
plt.xlabel("Label", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig("graphs/class_distribution.png", dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(14, 10))
correlation = df.drop(columns='label').corr()
sns.heatmap(correlation, cmap='viridis', center=0, square=True, linewidths=0.05, linecolor='black', cbar_kws={"shrink": 0.5})
plt.title("Feature Correlation Heatmap", fontsize=14, color='#08F7FE')
plt.savefig("graphs/feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()


variances = df.drop(columns='label').var().sort_values(ascending=False)
top10_features = variances.head(10).index
plt.figure(figsize=(8, 5))
sns.barplot(x=variances[top10_features].values, y=top10_features, palette=neon_colors)
plt.title("Top 10 Features by Variance", fontsize=14, color='#FE53BB')
plt.xlabel("Variance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.2)
plt.savefig("graphs/top10_feature_variance.png", dpi=300, bbox_inches='tight')
plt.show()


X = df.drop(columns='label')
y = df['label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['label'] = y

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='label', palette=neon_colors[:2], alpha=0.7, s=80, edgecolor='white', linewidth=0.4)
plt.title("PCA Projection (2D)", fontsize=14, color='#00ff41')
plt.legend(title='Label')
plt.grid(True, linestyle=':', alpha=0.2)
plt.savefig("graphs/pca_projection_2d.png", dpi=300, bbox_inches='tight')
plt.show()


key_features = ['feature_0', 'feature_1', 'feature_5', 'feature_10']
plt.figure(figsize=(12, 8))
for i, feat in enumerate(key_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='label', y=feat, data=df, palette=neon_colors)
    plt.title(f"{feat} by Label", fontsize=12, color='#F5D300')
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(True, linestyle='--', alpha=0.3)
plt.suptitle("Key Feature Distributions by Class", fontsize=15, color='#FE53BB')
plt.tight_layout()
plt.savefig("graphs/key_features_boxplot.png", dpi=300, bbox_inches='tight')
plt.show()


X = df.drop(columns='label')
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, stratify=y, random_state=1)

models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier()
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("\nBest Model:", best_model.__class__.__name__)
print(classification_report(y_test, best_model.predict(X_test)))
print(confusion_matrix(y_test, best_model.predict(X_test)))

y_best_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_best_pred))
print("Classification Report:\n", classification_report(y_test, y_best_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_best_pred))


input_data = (
    0.0453, 0.0523, 0.0843, 0.0689, 0.1183, 0.2583, 0.2156, 0.3481, 0.3337, 0.2872,
    0.4918, 0.6552, 0.6919, 0.7797, 0.7464, 0.9444, 1.0000, 0.8874, 0.8024, 0.7818,
    0.5212, 0.4052, 0.3957, 0.3914, 0.3250, 0.3200, 0.3271, 0.2767, 0.4423, 0.2028,
    0.3788, 0.2947, 0.1984, 0.2341, 0.1306, 0.4182, 0.3835, 0.1057, 0.1840, 0.1970,
    0.1674, 0.0583, 0.1401, 0.1628, 0.0621, 0.0203, 0.0530, 0.0742, 0.0409, 0.0061,
    0.0125, 0.0084, 0.0089, 0.0048, 0.0094, 0.0191, 0.0140, 0.0049, 0.0052, 0.0044
)
input_data_np = np.asarray(input_data).reshape(1, -1)
input_data_scaled = scaler.transform(input_data_np)
prediction = best_model.predict(input_data_scaled)

print(f"\nPrediction for the input data: {'Rock' if prediction[0] == 'R' else 'Mine'}")
