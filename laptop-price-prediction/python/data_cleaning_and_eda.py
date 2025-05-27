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

plt.style.use('dark_background')

plt.figure(figsize=(10, 6), dpi=300)
sns.histplot(df['Price'], kde=True, color='skyblue', bins=30, edgecolor='black')
plt.title('Distribution of Laptop Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('distributions/price_distribution.png', dpi=300)
plt.show()


plt.style.use('dark_background')

plt.figure(figsize=(12, 6), dpi=300)
df['Company'].value_counts().plot(kind='bar', color='orange')

plt.title('Laptop Count by Company')
plt.xlabel('Company')
plt.ylabel('Count')

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('distributions/company_count.png', dpi=300)
plt.show()

plt.style.use('dark_background')

plt.figure(figsize=(14, 6), dpi=300)
sns.barplot(x=df['Company'], y=df['Price'], estimator='mean', ci=None, color='lime', edgecolor='white')
plt.title('Average Laptop Price by Company')
plt.xlabel('Company')
plt.ylabel('Average Price')
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig('distributions/avg_price_by_company.png', dpi=300)
plt.show()

plt.style.use('dark_background')

plt.figure(figsize=(10, 6), dpi=300)
df['TypeName'].value_counts().plot(kind='bar', color='violet', edgecolor='white')
plt.title('Laptop Count by Type')
plt.xlabel('Type Name')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('distributions/type_count.png', dpi=300)
plt.show()

plt.style.use('dark_background')
plt.figure(figsize=(12, 6), dpi=300)
sns.barplot(x=df['TypeName'], y=df['Price'], estimator='mean', ci=None, color='teal')
plt.title('Average Laptop Price by Type')
plt.xlabel('Type Name')
plt.ylabel('Average Price')
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig('distributions/avg_price_by_type.png', dpi=300)
plt.show()


plt.style.use('dark_background')

plt.figure(figsize=(10, 6), dpi=300)
sns.histplot(df['Inches'], kde=True, color='magenta', edgecolor='white', bins=20)
plt.title('Distribution of Screen Sizes (Inches)', color='white')
plt.xlabel('Inches')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('distributions/inches_distribution.png', dpi=300)
plt.show()

plt.style.use('dark_background')

plt.figure(figsize=(10, 6), dpi=300)
sns.scatterplot(x=df['Inches'], y=df['Price'], color='cyan', edgecolor='white')
plt.title('Price vs Screen Size (Inches)', color='white')
plt.xlabel('Inches')
plt.ylabel('Price')
plt.tight_layout()
plt.savefig('distributions/price_vs_inches.png', dpi=300)
plt.show()

df['TouchScreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)

plt.style.use('dark_background')

plt.figure(figsize=(8, 5), dpi=300)
df['TouchScreen'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Count of TouchScreen Laptops')
plt.xlabel('TouchScreen')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('distributions/touchscreen_count.png', dpi=300)
plt.show()

plt.style.use('dark_background')
plt.figure(figsize=(8, 5), dpi=300)
sns.barplot(x=df['TouchScreen'], y=df['Price'], estimator='mean', ci=None, color='orchid')
plt.title('Average Price by TouchScreen Availability')
plt.xlabel('TouchScreen')
plt.ylabel('Average Price')
plt.tight_layout()
plt.savefig('distributions/avg_price_by_touchscreen.png', dpi=300)
plt.show()

df['Price'] = df['Price']*0.012

df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

plt.style.use('dark_background')

plt.figure(figsize=(8, 5), dpi=300)
df['IPS'].value_counts().plot(kind='bar', color='deepskyblue')
plt.title('Count of IPS Display Laptops')
plt.xlabel('IPS')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('distributions/ips_count.png', dpi=300)
plt.show()

plt.style.use('dark_background')
plt.figure(figsize=(8, 5), dpi=300)
sns.barplot(x=df['IPS'], y=df['Price'], estimator='mean', ci=None, color='mediumslateblue')
plt.title('Average Price by IPS Display')
plt.xlabel('IPS')
plt.ylabel('Average Price')
plt.tight_layout()
plt.savefig('distributions/avg_price_by_ips.png', dpi=300)
plt.show()

df['X_res'] = (df['ScreenResolution'].str.split('x', n=1, expand=True))[0]
df['y_res'] = (df['ScreenResolution'].str.split('x', n=1, expand=True))[1]

df['X_res'] = df['X_res'].str.replace(',' ,'').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0])

print(datacmp.get_detailed(df))

df['X_res'] = df['X_res'].astype('int')
df['y_res'] = df['y_res'].astype('int')

df['ppi'] = (((df['X_res']**2) + (df['y_res']**2))**0.5 / df['Inches']).astype('float')

print(df.corr(numeric_only=True)['Price'])

df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'y_res'], inplace=True)

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

plt.style.use('dark_background')

plt.figure(figsize=(10, 6), dpi=300)
df['CpuBrand'].value_counts().plot(kind='bar', color='tomato')
plt.title('Count of CPUs by Brand')
plt.xlabel('CPU Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('distributions/cpubrand_count.png', dpi=300)
plt.show()


plt.style.use('dark_background')
plt.figure(figsize=(12, 6), dpi=300)

sns.barplot(x=df['CpuBrand'], y=df['Price'], estimator='mean', ci=None, color='coral')
plt.title('Average Price by CPU Brand')
plt.xlabel('CPU Brand')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('distributions/avg_price_by_cpubrand.png', dpi=300)
plt.show()

df.drop(columns=['Cpu', 'CpuName'], inplace=True)

plt.style.use('dark_background')
plt.figure(figsize=(8, 5), dpi=300)

df['Ram'].value_counts().plot(kind='bar', color='mediumseagreen')
plt.title('Count of Laptops by RAM Size')
plt.xlabel('RAM')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('distributions/ram_count.png', dpi=300)
plt.show()


plt.style.use('dark_background')
plt.figure(figsize=(8, 5), dpi=300)

sns.barplot(x=df['Ram'], y=df['Price'], estimator='mean', ci=None, color='seagreen')
plt.title('Average Price by RAM Size')
plt.xlabel('RAM')
plt.ylabel('Average Price')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('distributions/avg_price_by_ram.png', dpi=300)
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

plt.style.use('dark_background')
plt.figure(figsize=(12, 6), dpi=300)
sns.barplot(x=df['GpuBrand'], y=df['Price'], estimator=np.median, ci=None, color='slateblue')
plt.title('Median Price by GPU Brand')
plt.xlabel('GPU Brand')
plt.ylabel('Median Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('distributions/median_price_by_gpubrand.png', dpi=300)
plt.show()

df.drop(columns=['Gpu'], inplace=True)

print(df['OpSys'].value_counts())

plt.style.use('dark_background')
plt.figure(figsize=(10, 6), dpi=300)
sns.barplot(x=df['OpSys'], y=df['Price'], estimator='mean', ci=None, color='steelblue')
plt.title('Average Price by Operating System')
plt.xlabel('Operating System')
plt.ylabel('Average Price')
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig('distributions/avg_price_by_opsys.png', dpi=300)
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

plt.style.use('dark_background')
plt.figure(figsize=(10, 6), dpi=300)
sns.barplot(x=df['os'], y=df['Price'], estimator='mean', ci=None, color='mediumvioletred')
plt.title('Average Price by OS')
plt.xlabel('OS')
plt.ylabel('Average Price')
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig('distributions/avg_price_by_os.png', dpi=300)
plt.show()


plt.style.use('dark_background')

plt.figure(figsize=(10, 6), dpi=300)
sns.histplot(df['Weight'], kde=True, color='gold', bins=20)
plt.title('Distribution of Laptop Weights')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('distributions/weight_distribution.png', dpi=300)
plt.show()


plt.style.use('dark_background')
plt.figure(figsize=(10, 6), dpi=300)

sns.scatterplot(x=df['Weight'], y=df['Price'], color='cyan')
plt.title('Price vs Laptop Weight')
plt.xlabel('Weight (kg)')
plt.ylabel('Price')
plt.tight_layout()
plt.savefig('distributions/price_vs_weight.png', dpi=300)
plt.show()

print(df.corr(numeric_only=True)['Price'])

print(df.corr(numeric_only=True))

plt.style.use('dark_background')
plt.figure(figsize=(10, 8), dpi=300)

sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=0.5, linecolor='gray')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('distributions/correlation_heatmap.png', dpi=300)
plt.show()


plt.style.use('dark_background')
plt.figure(figsize=(10, 6), dpi=300)

sns.histplot(np.log(df['Price']), kde=True, color='lime', bins=40)
plt.title('Log-Transformed Price Distribution')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('distributions/log_price_distribution.png', dpi=300)
plt.show()

df.to_csv('laptop-price-prediction/data/cleaned_laptop_data.csv', index=False)
