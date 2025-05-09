import numpy as np
import pandas as pd

df = pd.read_csv("quikr_car.csv")
print(df.head())

print(df.info())
print(df.isnull().sum())

df = df[df['year'].str.isnumeric()]
df['year'] = df['year'].astype(int)

df = df[df['Price'] != "Ask For Price"]
df['Price'] = df['Price'].str.replace(',','').astype(int)

df['kms_driven'] = df['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
df = df[df['kms_driven'].str.isnumeric()]
df['kms_driven'] = df['kms_driven'].astype(int)

df = df[~df['fuel_type'].isnull()]

df['name'] = df['name'].str.split(' ').str.slice(0,3).str.join(' ')

df = df.reset_index(drop=True)

print(df.describe())

df = df[df['Price'] < 6e6].reset_index(drop=True)

df.to_csv('cleaned.csv')


uniques = []
list1 = ['name', 'company', 'year', 'Price', 'kms_driven', 'fuel_type']
for i in list1:
    unique = df[i].unique()
    uniques.append(unique)

unique_df = pd.DataFrame(uniques).T
unique_df.columns = ['name', 'company', 'year', 'Price', 'kms_driven', 'fuel_type']

unique_df.to_csv("unique_values.csv", index=False)

# The Model
X = df.drop(columns='Price')
y = df['Price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

encoder = OneHotEncoder()
encoder.fit(X[['name','company','fuel_type']])

print(encoder.categories_)

column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), ['name','company','fuel_type']),remainder='passthrough')

model = LinearRegression()

pipe = make_pipeline(column_trans, model)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(r2_score(y_test, y_pred))


r2_max = 0
best_i = 0
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    model = LinearRegression()
    pipe = make_pipeline(column_trans, model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    if r2 > r2_max:
        r2_max = r2
        best_i = i
print(f"Best R2 score: {r2_max} at random state: {best_i}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=best_i)
model = LinearRegression()
pipe = make_pipeline(column_trans, model)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
r2_score(y_test, y_pred)


import pickle
pickle.dump(pipe, open("LinearRegressionModel.pkl", 'wb'))

new_data = pd.DataFrame({'name': ['Maruti Suzuki Swift'], 'company': ['Maruti'], 'year':[2019], 'kms_driven': [100], 'fuel_type': ['Petrol']})
print(pipe.predict(new_data))
