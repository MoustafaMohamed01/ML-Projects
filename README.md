# Machine Learning Projects

Welcome to my **Machine Learning Portfolio Repository**.
This repository showcases a growing collection of machine learning projects built using industry-standard tools and techniques. Each project is focused on solving real-world problems with proper data preprocessing, model selection, and performance evaluation.

---

## Repository Structure

```

ML-Projects/
│
├── sonar-object-classification/           # Binary classification of sonar signals
│   ├── sonar\_object\_classification.ipynb
│   ├── sonar\_object\_classification.py
│   ├── requirements.txt
│   ├── README.md
│   └── data/
│       └── sonar.all-data
│
├── Car-Price-Prediction/                  # Linear Regression + Streamlit app for car prices
│   ├── model\_1/
│   │   ├── Car\_Price\_Prediction.ipynb
│   │   ├── app.py
│   │   ├── Car\_Price\_Prediction.py
│   │   ├── LinearRegressionModel.pkl
│   │   └── data/
│   │       ├── unique\_values.csv
│   │       └── quikr\_car.csv
│   ├── requirements.txt
│   └── README.md
│
├── diabetes-prediction/                   # Classification using health features
│   ├── diabetes\_prediction.ipynb
│   ├── diabetes\_prediction.py
│   ├── requirements.txt
│   ├── README.md
│   ├── data/
│   │   └── diabetes.csv
│   └── images/
│       ├── diabetes\_countplot.png
│       ├── diabetes\_heatmap.png
│       └── diabetes\_pairplot.png
│
├── creditcard-fraud-detector/
│   ├── data/
│   │   └── README.md
│   ├── model/
│   │   └── xgboost\_fraud\_model.pkl
│   ├── plots/
│   │   ├── correlation\_heatmap.png
│   │   └── top10\_correlated\_features.png
│   ├── credit\_fraud\_detection.ipynb
│   ├── credit\_fraud\_detection.py
│   ├── requirements.txt
│   └── README.md
│
├── laptop-price-prediction/               # Laptop price prediction using regression + Streamlit app
│   ├── data/
│   │   ├── laptop\_data.csv
│   │   └── cleaned\_laptop\_data.csv
│   ├── distributions/
│   │   └── (visualization images)
│   ├── models/
│   │   └── best\_laptop\_price\_model.pkl
│   ├── python/
│   │   ├── data\_cleaning\_and\_eda.py
│   │   ├── train\_model.py
│   │   └── predict.py
│   ├── streamlit\_app/
│   │   ├── app.py
│   │   └── screenshot1.png
│   ├── requirements.txt
│   └── README.md
│
└── README.md                              # Main repository documentation

```

---

## Completed Projects

### 🔹 [Sonar Object Classification](./sonar-object-classification/)

> **Binary classification** of sonar signals to detect whether the object is a **Rock** or a **Mine** using **Logistic Regression**.

* **Algorithm**: Logistic Regression  
* **Dataset**: Sonar signals with 60 frequency-based features  
* **Libraries**: `scikit-learn`, `pandas`, `numpy`  
* **Highlights**:  
  * Feature scaling with `StandardScaler`  
  * Stratified train-test split  
  * Evaluation with accuracy, classification report, and confusion matrix  

---

### 🔹 [Car Price Prediction](./Car-Price-Prediction/)

> Predicts used car prices based on **model, company, year, mileage, and fuel type** using **Linear Regression**.  
> Includes a deployed **Streamlit web application** for interactive use.

* **Algorithm**: Linear Regression  
* **Dataset**: [Quikr Car Dataset](https://github.com/rajtilakls2510/car_price_predictor/blob/master/quikr_car.csv)  
* **Libraries**: `scikit-learn`, `pandas`, `numpy`, `streamlit`, `pickle`  
* **Highlights**:  
  * Data cleaning and preprocessing of messy real-world data  
  * Encoding categorical features and handling outliers  
  * Streamlit interface to input car details and predict prices  
  * Packaged with `requirements.txt` for easy setup  

---

### 🔹 [Diabetes Prediction](./diabetes-prediction/)

> A machine learning project to predict diabetes using the **Pima Indians Diabetes Dataset**.  
> Includes model comparison and visual analysis.

* **Algorithms**: SVM, Logistic Regression, Decision Tree, Random Forest  
* **Dataset**: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
* **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`  
* **Highlights**:  
  * Cleaned and preprocessed the dataset  
  * Dark-themed Seaborn visualizations (countplot, heatmap, pairplot)  
  * Model comparison with accuracy metrics  
  * Predicts new user input via script interface  

---

### 🔹 [Credit Card Fraud Detector](./creditcard-fraud-detector/)

> Detects fraudulent credit card transactions using multiple **supervised machine learning models** with EDA, visualization, and model evaluation.

* **Algorithms**: Logistic Regression, Random Forest, XGBoost, SVM, KNN  
* **Dataset**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
* **Libraries**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `datacmp`, `joblib`  
* **Highlights**:  
  * Visual EDA with correlation heatmaps and top features  
  * Comparison of 5 ML algorithms (XGBoost: **99.96% accuracy**)  
  * Exported best-performing model (`xgboost_fraud_model.pkl`)  
  * Includes Jupyter notebook and standalone Python script  

---

### 🔹 [Laptop Price Prediction](./laptop-price-prediction/)

> Predicts laptop prices based on specifications like **brand, RAM, storage, CPU, GPU, screen size/resolution, and more** using multiple regression models.  
> Includes extensive EDA, feature engineering, and a **Streamlit web app** for real-time price predictions.

* **Algorithms**: Linear Regression, Random Forest, XGBoost, and more  
* **Dataset**: Raw and cleaned laptop specification data  
* **Libraries**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `streamlit`, `joblib`  
* **Highlights**:  
  * Detailed data cleaning and feature engineering (e.g., Pixels Per Inch from resolution)  
  * Visualization of price distributions and feature impacts  
  * Comparison and evaluation of multiple regression models  
  * Deployed Streamlit app with interactive input and prediction  
  * Well-structured project with modular Python scripts and saved model  

---

## Tech Stack

* **Languages**: Python 3.7+  
* **Libraries**:  
  `pandas`, `numpy`, `scikit-learn`, `streamlit`, `pickle`, `matplotlib`, `seaborn`, `datacmp`, `joblib`, `xgboost`  
* **Tools**:  
  Jupyter Notebook, Visual Studio Code, Git, GitHub, [DataCmp](https://pypi.org/project/datacmp/)  

---

## How to Contribute

Have an idea, bug fix, or feature suggestion?  
You're welcome to fork this repository, make improvements, and submit a pull request!

---

## Support

If you find these projects helpful or insightful, please consider giving the repo a **star**.  
Your support motivates continued learning and sharing!

---
## About Me

**Moustafa Mohamed**  
Aspiring AI Developer with a focus on **Machine Learning, Deep Learning**, and **LLM Engineering**.

* **GitHub**: [MoustafaMohamed01](https://github.com/MoustafaMohamed01)
* **Linkedin**: [Moustafa Mohamed](https://www.linkedin.com/in/moustafamohamed01/)
* **Kaggle**: [moustafamohamed01](https://www.kaggle.com/moustafamohamed01)
* **Portfolio**: [moustafamohamed](https://moustafamohamed.netlify.app/)

---
