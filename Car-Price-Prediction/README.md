# Car Price Prediction Project

## Overview
This repository contains a **Car Price Prediction** model that estimates the price of a used car based on various features such as **name, company, year, kilometers driven, and fuel type**. This model uses **Linear Regression** and is deployed as a **Streamlit web application**.

## Dataset
The dataset used in this project is **quikr_car.csv**, sourced from:
[Car Price Predictor Dataset](https://github.com/rajtilakls2510/car_price_predictor/blob/master/quikr_car.csv)

The dataset contains car listings with details such as:
- **Name** (Car Model)
- **Company** (Brand)
- **Year** (Manufacturing Year)
- **Price** (Car Price)
- **Kilometers Driven** (Distance the car has been driven)
- **Fuel Type** (Petrol/Diesel/CNG/Electric)

---

## File Structure
```
Car-Price-Prediction/
│
├── model_1/                          # Folder containing files for Model 1
│   ├── Car_Price_Prediction.ipynb    # Jupyter notebook for data cleaning, preprocessing, and model training
│   ├── app.py                        # Streamlit app for car price prediction
│   ├── Car_Price_Prediction.py       # Python script for implementing the model
│   ├── LinearRegressionModel.pkl     # Pickled model for predictions
│   ├── requirements.txt              # List of Python dependencies for the project
│   └── data/                         
│       ├── unique_values.csv             # File with unique values for encoding categorical data
│       └── quikr_car.csv                 # Original dataset for Model 1 preprocessing
│
├── requirements.txt                  # List of Python dependencies for the project
└── README.md                         # Main project README
```

---

## Install dependencies
1. Clone the repository:
   ```bash
   git clone https://github.com/MoustafaMohamed01/Car-Price-Prediction.git
   cd Car-Price-Prediction
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Run the Model

### **Model 1: Linear Regression**

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook
   ```
2. Open and run the `Car_Price_Prediction.ipynb` notebook for data preprocessing, model training, and evaluation.

#### **Streamlit Web App for Model 1**
To run the web app for Model 1, use the following command:
```bash
streamlit run model_1/app.py
```
This will launch the Streamlit app, where you can input car details and get a predicted price.

---

## File Descriptions

### **Model 1:**

- **Car_Price_Prediction.ipynb**: Jupyter notebook for data preprocessing, model training, and evaluation using **Linear Regression**.
- **app.py**: Streamlit app for **Model 1**, allowing users to interact with the model and make predictions.
- **Car_Price_Prediction.py**: Python script for implementing the model and training using **Linear Regression**.
- **LinearRegressionModel.pkl**: The trained **Linear Regression** model file (saved using **Pickle**) that is used for making predictions.
- **unique_values.csv**: Contains the unique values of categorical columns for encoding.
- **quikr_car.csv**: The **original dataset** for Model 1.
- **cleaned.csv**: The **cleaned version** of the dataset after preprocessing.

---

## How to Contribute

Feel free to fork this repository, make changes, and submit pull requests. If you have improvements or suggestions, please open an issue or create a pull request.
