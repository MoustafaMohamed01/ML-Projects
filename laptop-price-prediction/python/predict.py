import joblib
import numpy as np
import pandas as pd

model = joblib.load('laptop-price-prediction/models/best_laptop_price_model.pkl')

sample = pd.DataFrame([{
    'Company': 'Dell',
    'TypeName': 'Notebook',
    'Ram': 8,
    'Weight': 1.8,
    'TouchScreen': 0,
    'IPS': 1,
    'ppi': 141.21,
    'CpuBrand': 'Intel Core i5',
    'SSD': 512,
    'HDD': 0,
    'GpuBrand': 'Intel',
    'os': 'Windows 10'
}])

log_price = model.predict(sample)[0]
predicted_price = np.exp(log_price)

print(f"Predicted Laptop Price: ${predicted_price:.2f}")

