# Stock Price Movement Prediction using KNN

This project uses historical stock data of **TATA Global** to predict whether the stock's closing price will increase or decrease the next day using the **K-Nearest Neighbors (KNN)** classification algorithm.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Seaborn & Matplotlib (for visualization)
* Scikit-learn (for modeling)
* Quandl (for financial data access)
* [DataCmp](https://pypi.org/project/datacmp/) (for quick data analysis)

---

## Dataset

The dataset used is named `NSE-TATAGLOBAL11.csv` and contains historical stock prices with features like:

* `Open`
* `Close`
* `High`
* `Low`

---

## Features Engineered

Two new features were created:

* `Open - Close`: Difference between opening and closing price
* `High - Low`: Difference between daily high and low

---

## Target Variable

The target is to **predict the stock's movement**:

* `1` if the next day's `Close` > current `Close`
* `-1` otherwise

---

## Model Training

* Model used: **K-Nearest Neighbors (KNN)**
* Hyperparameter tuning: `GridSearchCV` over different `n_neighbors` values
* Evaluation: **Train and Test Accuracy**

---

## Results

After training:

* **Train Accuracy**: \~`0.88`
* **Test Accuracy**: \~`0.86`

> You can expect the model to give a general sense of stock direction, but it's not financial advice.

---

## How to Run

1. Clone this repo.
2. Ensure the CSV file `NSE-TATAGLOBAL11.csv` is in the project folder.
3. Install dependencies:

   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn quandl datacmp
   ```
4. Run the notebook or Python script.

---

## Note

This project is for **educational purposes only** and should not be used for real financial decisions.
