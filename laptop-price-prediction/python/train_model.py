import numpy as np
import pandas as pd
import datacmp
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

df = pd.read_csv('laptop-price-prediction/data/cleaned_laptop_data.csv')
print(df.head())

print(datacmp.get_detailed(df))

X = df.drop(columns=['Price'])
y = np.log(df['Price'])

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

joblib.dump(best_model, 'laptop-price-prediction/models/best_laptop_price_model.pkl')

print("Model saved as 'best_laptop_price_model.pkl'")
