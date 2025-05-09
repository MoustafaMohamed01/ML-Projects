# Car Price Prediction

## Overview
This project is a **Car Price Prediction** model that estimates the price of a used car based on various features like **name, company, year, kilometers driven, and fuel type**. The model is built using **Linear Regression** and is deployed as a **Streamlit web application**.

## Dataset
The dataset used in this project is `quikr_car.csv`, sourced from:
[Car Price Predictor Dataset](https://github.com/rajtilakls2510/car_price_predictor/blob/master/quikr_car.csv)

It contains car listings with details such as:
- **Name** (Car Model)
- **Company** (Brand)
- **Year** (Manufacturing Year)
- **Price** (Car Price)
- **Kilometers Driven** (Distance the car has been driven)
- **Fuel Type** (Petrol/Diesel/LPG)

## Files in the Repository
- `Car_Price_Prediction.ipynb` – Jupyter Notebook for data cleaning, preprocessing, model training, and evaluation.
- `app.py` – Python script for the Streamlit web app to interact with the model.
- `Car_Price_Prediction.py` – Python script containing the main model implementation.
- `LinearRegressionModel.pkl` – Trained Linear Regression model saved using Pickle.
- `unique_values.csv` – Contains unique values of categorical columns used for encoding.
- `quikr_car.csv` – Original dataset.

## Data Preprocessing
The dataset undergoes the following preprocessing steps:
1. **Cleaning:**
   - Removing non-numeric values from columns.
   - Removing outliers in **Price**.
   - Handling missing values.
   - Converting **year** and **kms_driven** to integers.
   - Standardizing car **name** (keeping first three words).
2. **Encoding Categorical Data:**
   - One-Hot Encoding applied to **name, company, and fuel type**.
3. **Feature Selection & Splitting:**
   - Target variable: `Price`
   - Features: `name, company, year, kms_driven, fuel_type`
   - Splitting into training and testing sets (80-20 split).

## Model Training
- **Algorithm:** Linear Regression
- **Pipeline:**
  - One-Hot Encoding for categorical features.
  - Applying **Linear Regression** to predict car price.
- **Evaluation:**
  - Using **R² Score** to measure model accuracy.
  - Finding the best **random state** for train-test split to maximize R².

## Model Deployment (Streamlit App)
The model is integrated into a **Streamlit** web app where users can input car details and get a price prediction.

### **Run the Web App**
To run the web app, execute:
```bash
streamlit run app.py
```

## Installation & Setup
### **1. Clone the Repository**
```bash
git clone https://github.com/MoustafaMohamed01/Car-Price-Prediction.git
cd Car-Price-Prediction\model_1
```

### **2. Install Required Packages**
```bash
pip install -r requirements.txt
```

### **3. Run the Jupyter Notebook**
```bash
jupyter notebook
```
Open `Car_Price_Prediction.ipynb` and run all cells.

