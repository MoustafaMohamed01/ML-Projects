# Sonar Object Classification using Machine Learning

This project applies a range of supervised machine learning algorithms to classify sonar signals as either **mines** or **rocks** based on sonar return strengths. The dataset contains 60 continuous features representing energy patterns from sonar returns.  

Through exploratory analysis, dimensionality reduction, and rigorous model comparison, this project identifies the most accurate classifier for real-time prediction scenarios.

---

## Directory Structure

```

sonar-object-classification/
├── data/                             # Dataset (raw or preprocessed)
├── graphs/                           # Visualizations (dark-themed, high-resolution)
├── sonar_object_classification.ipynb # Jupyter notebook with full workflow
├── sonar_object_classification.py    # Python script version of the notebook
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation

```

---

## Dataset Overview

- **Source**: [Sonar Dataset](https://www.kaggle.com/datasets/mahmudulhaqueshawon/sonar-data)
- **Instances**: 208
- **Features**: 60 numeric attributes (frequency-based energy values)
- **Target**: `M` (Mine), `R` (Rock)

---

## Exploratory Data Analysis (EDA)

EDA was performed using dark-themed, high-resolution visualizations for clarity and presentation. Key visualizations include:

- **Class Distribution** – Verifying data balance
- **Feature Correlation Heatmap** – Identifying inter-feature relationships
- **Top 10 Features by Variance** – Highlighting the most informative features
- **PCA (2D Projection)** – Visualizing data separability
- **Boxplots** – Comparing feature distributions across classes

> All figures are saved in the `graphs/` directory with `dpi=300`.

---

## Model Training and Evaluation

Multiple classification algorithms were trained and evaluated using standardized data. A stratified train-test split (90/10) was applied to ensure balanced representation across classes.

### Models Evaluated

| Model                     | Accuracy   |
|--------------------------|------------|
| Logistic Regression       | 76.19%     |
| K-Nearest Neighbors (KNN) | **90.48%** |
| Support Vector Machine    | 80.95%     |
| Decision Tree             | 80.95%     |
| Random Forest             | 71.43%     |
| Gradient Boosting         | 76.19%     |
| Naive Bayes               | 61.90%     |

### Best Model: **K-Nearest Neighbors (KNN)**

**Performance on Test Set:**

- **Accuracy**: 90.48%
- **Precision**: 85% (Mine), 100% (Rock)
- **Recall**: 100% (Mine), 80% (Rock)
- **F1-Score**: 0.92 (Mine), 0.89 (Rock)

---

## Real-Time Prediction

A prediction pipeline was implemented to classify new, unseen sonar readings.  
The input data is preprocessed using the same scaler, then passed to the trained best model (`KNeighborsClassifier`) for inference.

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/MoustafaMohamed01/ML-Projects.git
cd ML-Projects/sonar-object-classification
````

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Notebook

Open and execute:

```
sonar_object_classification.ipynb
```

Or run the script:

```bash
python sonar_object_classification.py
```

---

## Author

**Moustafa Mohamed**

Aspiring AI Developer | Machine Learning & Deep Learning & LLMs Specialist

[GitHub](https://github.com/MoustafaMohamed01) • [Kaggle](https://www.kaggle.com/moustafamohamed01) • [LinkedIn](https://www.linkedin.com/in/moustafamohamed01/)

---
