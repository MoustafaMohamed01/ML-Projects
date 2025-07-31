# Machine Learning Projects

Welcome to my **Machine Learning Portfolio Repository**.

This repository contains a growing collection of machine learning projects that solve real-world problems using industry-standard tools, clean code practices, and comprehensive data workflows — from preprocessing and feature engineering to model training, evaluation, and deployment.

---

## Repository Structure

```

ML-Projects/
│
├── sonar-object-classification/           # Sonar signal classification (Mine vs. Rock)
├── Car-Price-Prediction/                  # Regression + Streamlit app for car pricing
├── diabetes-prediction/                   # Disease classification based on health metrics
├── creditcard-fraud-detector/             # Anomaly detection using XGBoost and others
├── laptop-price-prediction/               # Regression + Streamlit app for laptops
├── stock-price-prediction/                # Stock movement classifier
├── loan-status-prediction/                # Predicting loan approvals
└── README.md                              # Main repository documentation

```

---

## Completed Projects

### 🔹 [Sonar Object Classification](./sonar-object-classification/)

> **Binary classification** of sonar signals to detect whether the object is a **mine** or a **rock** using a range of supervised machine learning models.

* **Algorithms**: Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Naive Bayes, Gradient Boosting  
* **Dataset**: [UCI Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench/sonar)
* **Libraries**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Highlights**:
  - Feature scaling with `StandardScaler`
  - Principal Component Analysis (PCA) for 2D visualization
  - Comparison of 7 classifiers
  - Best model: **K-Nearest Neighbors** (90.48% accuracy)
  - Dark-themed visualizations (saved in `graphs/` folder)

---

### 🔹 [Car Price Prediction](./Car-Price-Prediction/)

> Predicts used car prices using a **Linear Regression** model with a **Streamlit web interface**.

* **Algorithm**: Linear Regression
* **Libraries**: `scikit-learn`, `pandas`, `numpy`, `streamlit`, `pickle`
* **Highlights**:
  - Feature engineering and data cleaning
  - Deployment-ready interface for real-time prediction
  - Modular scripts and saved model

---

### 🔹 [Diabetes Prediction](./diabetes-prediction/)

> Predicts diabetes likelihood using health-related metrics with multiple classifiers.

* **Algorithms**: SVM, Logistic Regression, Random Forest, Decision Tree
* **Dataset**: Pima Indians Diabetes Dataset
* **Highlights**:
  - Dark-mode visualizations
  - Model comparison and metric evaluation
  - User input prediction via script

---

### 🔹 [Credit Card Fraud Detection](./creditcard-fraud-detector/)

> Detects fraudulent transactions using advanced classification models including **XGBoost**.

* **Algorithms**: XGBoost, Logistic Regression, SVM, Random Forest, KNN
* **Highlights**:
  - EDA with `DataCmp` and visualizations
  - XGBoost achieved **99.96% accuracy**
  - Exported model for deployment

---

### 🔹 [Laptop Price Prediction](./laptop-price-prediction/)

> Predicts laptop prices using regression models and delivers results via a **Streamlit app**.

* **Features**: Screen resolution, CPU, GPU, RAM, storage, and more
* **Highlights**:
  - Rich EDA and feature engineering
  - Deployment-ready with modular code
  - Comparison of regression models

---

### 🔹 [Stock Price Prediction](./stock-price-prediction/)

> Predicts next-day stock movement using engineered financial features and ML classifiers.

* **Algorithms**: SVM, Random Forest, KNN, XGBoost, etc.
* **Highlights**:
  - Binary classification (Up/Down)
  - Accuracy heatmap comparison
  - GridSearchCV tuning

---

### 🔹 [Loan Status Prediction](./loan-status-prediction/)

> Predicts loan approval status based on financial and personal attributes.

* **Algorithm**: Support Vector Machine (Linear Kernel)
* **Highlights**:
  - Handling of missing values and categorical encoding
  - Clean visualizations and performance metrics
  - Well-structured scripts and notebook

---

## Tech Stack

- **Languages**: Python 3.7+
- **ML Libraries**: `scikit-learn`, `xgboost`, `joblib`, `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `streamlit`
- **Other Tools**: Jupyter Notebook, Git, VSCode, Kaggle, DataCmp

---

## About Me

**Moustafa Mohamed**  
Aspiring AI Developer | Focused on Machine Learning, Deep Learning & LLM Engineering

* **GitHub**: [MoustafaMohamed01](https://github.com/MoustafaMohamed01)
* **Linkedin**: [Moustafa Mohamed](https://www.linkedin.com/in/moustafamohamed01/)
* **Kaggle**: [moustafamohamed01](https://www.kaggle.com/moustafamohamed01)
* **Portfolio**: [moustafamohamed](https://moustafamohamed.netlify.app/)

---

## ⭐ Contribute & Support

If you find this repository helpful, please consider giving it a ⭐.  
Contributions, feedback, and ideas are always welcome!

---
