import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datacmp

df = pd.read_csv('laptop-price-prediction/data/laptop_data.csv')
print(df.head())

print(datacmp.get_detailed(df))
print(df.duplicated().sum())

df.drop(columns=['Unnamed: 0'], inplace=True)

df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')

df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = pd.to_numeric(df['Weight'])
