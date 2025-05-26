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

