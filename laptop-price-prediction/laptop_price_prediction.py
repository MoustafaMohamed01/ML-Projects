import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datacmp

df = pd.read_csv('laptop_data.csv')
df.head()

print(datacmp.get_detailed(df))
print(df.duplicated().sum())

df.drop(columns=['Unnamed: 0'], inplace=True)

df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')

df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = pd.to_numeric(df['Weight'])

sns.distplot(df['Price'])
plt.show()

df['Company'].value_counts().plot(kind='bar')
plt.show()

sns.barplot(x=df['Company'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

df['TypeName'].value_counts().plot(kind='bar')
plt.show()

sns.barplot(x=df['TypeName'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.distplot(df['Inches'])
plt.show()

sns.scatterplot(x=df['Inches'], y=df['Price'])
plt.show()

df['TouchScreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)

df['TouchScreen'].value_counts().plot(kind='bar')
plt.show()

sns.barplot(x= df['TouchScreen'], y= df['Price'])
plt.show()

df['Price'] = df['Price']*0.012
df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

df['IPS'].value_counts().plot(kind='bar')
plt.show()

sns.barplot(x=df['IPS'], y=df['Price'])
plt.show()

df['X_res'] = (df['ScreenResolution'].str.split('x', n=1, expand=True))[0]
df['y_res'] = (df['ScreenResolution'].str.split('x', n=1, expand=True))[1]
df['X_res'] = df['X_res'].str.replace(',' ,'').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0])

print(datacmp.get_detailed(df))

df['X_res'] = df['X_res'].astype('int')
df['y_res'] = df['y_res'].astype('int')

df['ppi'] = (((df['X_res']**2) + (df['y_res']**2))**0.5 / df['Inches']).astype('float')
print(df.corr(numeric_only=True)['Price'])

df.drop(columns=['ScreenResolution'], inplace=True)
df.drop(columns=['Inches', 'X_res', 'y_res'], inplace=True)

df['CpuName'] = df['Cpu'].apply(lambda x: ' '.join(x.split()[0:3]))

def get_cpu_brand(cpu_name):
    if cpu_name == 'Intel Core i7' or cpu_name == 'Intel Core i5' or cpu_name == 'Intel Core i3':
        return cpu_name
    elif 'Intel' in cpu_name:
        return 'Intel'
    elif 'AMD' in cpu_name:
        return 'AMD'
    elif 'Samsung' in cpu_name:
        return 'Samsung'
    else:
        return 'Other'

df['CpuBrand'] = df['CpuName'].apply(get_cpu_brand)
print(df['CpuBrand'].value_counts())

df = df[df['CpuBrand'] != 'Samsung']

df['CpuBrand'].value_counts().plot(kind='bar')
plt.show()

sns.barplot(x=df['CpuBrand'], y=df['Price'])
plt.show()

df.drop(columns=['Cpu', 'CpuName'], inplace=True)

df['Ram'].value_counts().plot(kind='bar')
plt.show()

sns.barplot(x=df['Ram'],y=df['Price'])
plt.show()

print(df['Memory'].value_counts())

df['Memory'] = df['Memory'].str.replace(r'\s+', ' ', regex=True).str.strip().str.upper()

df['SSD'] = 0
df['HDD'] = 0
df['Flash'] = 0
df['Hybrid'] = 0

def convert_to_gb(size_str):
    if 'TB' in size_str:
        return int(float(size_str.replace('TB', '')) * 1024)
    elif 'GB' in size_str:
        return int(float(size_str.replace('GB', '')))
    return 0

for i, row in df.iterrows():
    parts = row['Memory'].split('+')
    for part in parts:
        part = part.strip()
        if 'SSD' in part and 'FLASH' not in part:
            df.at[i, 'SSD'] += convert_to_gb(part.replace('SSD', '').strip())
        elif 'HDD' in part:
            df.at[i, 'HDD'] += convert_to_gb(part.replace('HDD', '').strip())
        elif 'FLASH' in part:
            df.at[i, 'Flash'] += convert_to_gb(part.replace('FLASH STORAGE', '').strip())
        elif 'HYBRID' in part:
            df.at[i, 'Hybrid'] += convert_to_gb(part.replace('HYBRID', '').strip())

df.drop('Memory', axis=1, inplace=True)

print(df.corr(numeric_only=True)['Price'])

df.drop(columns=['Hybrid', 'Flash'], inplace=True)

print(df['Gpu'].value_counts())

df['GpuBrand'] = df['Gpu'].apply(lambda x:x.split()[0])

print(df['GpuBrand'].value_counts())

sns.barplot(x=df['GpuBrand'], y=df['Price'], estimator=np.median)
plt.show()

df.drop(columns=['Gpu'], inplace=True)
print(df['OpSys'].value_counts())

sns.barplot(x=df['OpSys'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

def clean_opsys(os):
    os = os.strip().upper()
    if 'WINDOWS' in os:
        return 'Windows'
    elif 'MAC' in os or 'MACOS' in os:
        return 'macOS'
    elif 'LINUX' in os:
        return 'Linux'
    elif 'CHROME' in os:
        return 'Chrome OS'
    elif 'ANDROID' in os:
        return 'Android'
    elif 'NO OS' in os:
        return 'No OS'
    else:
        return 'Other'

df['os'] = df['OpSys'].apply(clean_opsys)
df.drop(columns=['OpSys'], inplace=True)

sns.barplot(x=df['os'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.distplot(df['Weight'])
plt.show()

sns.scatterplot(x=df['Weight'], y=df['Price'])
plt.show()

print(df.corr(numeric_only=True)['Price'])
print(df.corr(numeric_only=True))

sns.heatmap(df.corr(numeric_only=True))
plt.show()

sns.distplot(np.log(df['Price']))
plt.show()

X = df.drop(columns=['Price'])
y = np.log(df['Price'])

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(exclude='object').columns.tolist()

preprocessor = ColumnTransformer(
    [
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('scale', StandardScaler(), numerical_cols)
    ]
)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'KNN': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'Extra Trees': ExtraTreesRegressor(),
    'SVR': SVR(),
    'XGBoost': XGBRegressor()
}

results = []

for name, model in models.items():
    pipe = Pipeline(
        [
            ('preprocessing', preprocessor),
            ('regressor', model)
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append(
        {
            'Model': name,
            'R2 Score': r2,
            'MAE': mae,
            'RMSE': rmse
        }
    )

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='R2 Score', ascending=False).reset_index(drop=True)
print(results_df)

best_model = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', SVR())
])

best_model.fit(X, y)

joblib.dump(best_model, 'laptop_price_model.pkl')

print("Model saved as 'best_laptop_price_model.pkl'")


model = best_model

sample = pd.DataFrame([{
    'Company': 'Dell',
    'TypeName': 'Notebook',
    'Ram': 8,
    'Weight': 1.8,
    'TouchScreen': 0,
    'IPS': 1,
    'ppi': 141.21,
    'CpuBrand': 'Intel Core i5',
    'SSD': 256,
    'HDD': 0,
    'GpuBrand': 'Intel',
    'os': 'Windows 10'
}])

log_price = model.predict(sample)[0]
predicted_price = np.exp(log_price)

print(f"Predicted Laptop Price: ${predicted_price:.2f}")

