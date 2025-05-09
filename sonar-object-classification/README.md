# Sonar Object Classification

This project uses **Logistic Regression** to classify sonar signals as either **Rock** or **Mine** based on the reflection of sonar pulses. The model is trained on the [UCI Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+sonar+mines+vs.+rocks), which contains 60 numerical features extracted from sonar signals.

---

## Project Structure

```

sonar-object-classification/
│
├── data/
│   └── sonar.all-data                 # Raw sonar dataset (no header)
│
├── sonar_object_classification.ipynb  # Jupyter notebook version of the project
├── sonar_object_classification.py     # Python script version of the project
├── requirements.txt
└── README.md                          # Project documentation
````

---

## Model Overview

- **Algorithm**: Logistic Regression
- **Preprocessing**: Feature scaling with `StandardScaler`
- **Train/Test Split**: 90% training, 10% testing (stratified)
- **Evaluation Metrics**: Accuracy, Classification Report, Confusion Matrix

---

## How It Works

1. Load and prepare the sonar dataset.
2. Preprocess features using standard scaling.
3. Train a Logistic Regression model.
4. Evaluate performance on training and testing sets.
5. Make a prediction on new input data (60 features).
6. Output whether the object is a **Rock** or **Mine**.

---

## Dependencies

* Python 3.7+
* pandas
* numpy
* scikit-learn
* datacmp

Install dependencies via pip:

```bash
pip install pandas numpy scikit-learn datacmp
```
or:
```bash
pip install -r requirements.txt
```

---

## Dataset Source

The dataset can be downloaded from the UCI ML Repository:
 [https://archive.ics.uci.edu/ml/datasets/connectionist+bench+sonar+mines+vs.+rocks](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+sonar+mines+vs.+rocks)

---

## Author

**Moustafa Mohamed**
 [LinkedIn](https://www.linkedin.com/in/moustafa-mohamed-047736296/)
| [GitHub](https://github.com/MoustafaMohamed01)
| [Kaggle](https://www.kaggle.com/moustafamohamed01)
