import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def set_dark_style():
    plt.style.use('dark_background')

    plt.rcParams.update({
        'axes.labelcolor': 'cyan',
        'xtick.color': 'cyan',
        'ytick.color': 'cyan',
        'text.color': 'cyan',
        'figure.facecolor': '#121212',
        'axes.facecolor': '#121212',
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.alpha': 0.4,
        'font.family': 'sans-serif'
    })

def ensure_images_folder():
    if not os.path.exists('images'):
        os.makedirs('images')

def load_data(filepath: str):
    df = pd.read_csv(filepath)
    return df

def explore_data(df: pd.DataFrame):
    print("First 5 rows:")
    print(df.head())

    print("\nDataframe Info:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nColumn names:")
    print(df.columns)

    set_dark_style()
    ensure_images_folder()

    plt.figure(dpi=300)
    sns.countplot(x='quality', data=df, palette='coolwarm')
    plt.title('Distribution of Wine Quality', fontsize=14, color='#00FFCC')
    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/quality_distribution.png', dpi=300)
    plt.show()

    plt.figure(figsize=(8, 6), dpi=300)
    corr = df.corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        square=True,
        linewidths=1,
        linecolor='#00FFCC',
        vmin=-1,
        vmax=1,
        center=0,
        annot_kws={'color': '#00FFCC'}
    )
    plt.title('Feature Correlation Heatmap', fontsize=14, color='#00FFCC')
    plt.tight_layout()
    plt.savefig('images/correlation_heatmap.png', dpi=300)
    plt.show()

def prepare_data(df: pd.DataFrame):
    X = df.drop('quality', axis=1)
    y = df['quality'].apply(lambda q: 1 if q >= 7 else 0)
    print("\nTarget classes and counts:")
    print(y.value_counts())
    return X, y

def split_data(X, y, test_size=0.2, random_state=2):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

def plot_model_accuracies(model_accuracies):
    set_dark_style()
    ensure_images_folder()
    plt.figure(figsize=(8, 5), dpi=300)
    names = list(model_accuracies.keys())
    accuracies = list(model_accuracies.values())

    bars = plt.bar(names, accuracies, color='#00FFCC', edgecolor='#00FFCC', alpha=0.8)
    plt.ylim(0, 1)
    plt.ylabel('Accuracy', fontsize=12, color='#00FFCC')
    plt.title('Model Accuracy Comparison', fontsize=14, color='#00FFCC')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height - 0.07, f'{acc:.3f}',
                 ha='center', color='#121212', fontsize=11, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.3, color='#00FFCC')
    plt.tight_layout()
    plt.savefig('images/model_accuracy_comparison.png', dpi=300)
    plt.show()

def compare_models_with_plot(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(random_state=2),
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=2),
        'SVM (RBF Kernel)': SVC(kernel='rbf', random_state=2)
    }

    model_accuracies = {}
    best_acc = 0
    best_model = None
    best_name = ''

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {acc:.4f}")
        model_accuracies[name] = acc
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    plot_model_accuracies(model_accuracies)

    print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")
    return best_model, best_name, best_acc

def predict_quality(model, input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    print(f"\nPrediction for input data: {prediction[0]}")
    if prediction[0] == 1:
        print("Result: Good Quality Wine")
    else:
        print("Result: Bad Quality Wine")

df = load_data('winequality-red.csv')
explore_data(df)

X, y = prepare_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)

best_model, best_name, best_acc = compare_models_with_plot(X_train, X_test, y_train, y_test)

sample_input = (7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4)
predict_quality(best_model, sample_input)
