
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train_data = pd.read_csv('data/train_data.csv')
test_data = pd.read_csv('data/test_data.csv')

"""# Data overview"""



train_no_duplicates = train_data.drop_duplicates()


test_no_duplicates = test_data.drop_duplicates()


train_cleaned = train_no_duplicates[train_no_duplicates['ID'].notna()].copy()


test_cleaned = test_no_duplicates[test_no_duplicates['ID'].notna()].copy()


train_cleaned['Prod. year'] = train_cleaned['Prod. year'].fillna(train_cleaned['Prod. year'].mode()[0])


test_cleaned['Prod. year'] = test_cleaned['Prod. year'].fillna(test_cleaned['Prod. year'].mode()[0])


train_cleaned.loc[train_data['Model'].notna(), 'Manufacturer'] = (
    train_cleaned.loc[train_data['Model'].notna()]
    .groupby('Model')['Manufacturer']
    .transform(
        lambda x: x.fillna(
            x.mode().iloc[0] if not x.mode().empty else train_data['Manufacturer'].mode().iloc[0]
        )
    )
)


test_cleaned.loc[test_data['Model'].notna(), 'Manufacturer'] = (
    test_cleaned.loc[test_data['Model'].notna()]
    .groupby('Model')['Manufacturer']
    .transform(
        lambda x: x.fillna(
            x.mode().iloc[0] if not x.mode().empty else test_data['Manufacturer'].mode().iloc[0]
        )
    )
)


train_cleaned.loc[train_cleaned['Manufacturer'].notna(), 'Model'] = (
    train_cleaned.loc[train_cleaned['Manufacturer'].notna()]
    .groupby('Manufacturer')['Model']
    .transform(
        lambda x: x.fillna(
            x.mode().iloc[0] if not x.mode().empty else train_cleaned['Model'].mode().iloc[0]
        )
    )
)


test_cleaned.loc[test_cleaned['Manufacturer'].notna(), 'Model'] = (
    test_cleaned.loc[test_cleaned['Manufacturer'].notna()]
    .groupby('Manufacturer')['Model']
    .transform(
        lambda x: x.fillna(
            x.mode().iloc[0] if not x.mode().empty else test_cleaned['Model'].mode().iloc[0]
        )
    )
)


train_cleaned['Fuel type'] = (
    train_cleaned.groupby('Manufacturer')['Fuel type']
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))
)

print(train_cleaned['Fuel type'].isnull().sum())

test_cleaned['Fuel type'] = (
    test_cleaned.groupby('Manufacturer')['Fuel type']
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))
)


train_cleaned['Gear box type'] = (
    train_cleaned.groupby('Manufacturer')['Gear box type']
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))
)


test_cleaned['Gear box type'] = (
    test_cleaned.groupby('Manufacturer')['Gear box type']
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))
)


train_cleaned['Engine volume'] = (
    train_cleaned.groupby('Model')['Engine volume']
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else train_cleaned['Engine volume'].mode().iloc[0]))
)


test_cleaned['Engine volume'] = (
    test_cleaned.groupby('Model')['Engine volume']
    .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else test_cleaned['Engine volume'].mode().iloc[0]))
)


train_cleaned['Mileage'] = (
    train_cleaned.groupby('Model')['Mileage']
     .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else train_cleaned['Mileage'].mode().iloc[0]))
)


test_cleaned['Mileage'] = (
    test_cleaned.groupby('Model')['Mileage']
     .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else test_cleaned['Mileage'].mode().iloc[0]))
)


train_cleaned['Leather interior'] = (
    train_cleaned.groupby('Model')['Leather interior']
     .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else train_cleaned['Leather interior'].mode().iloc[0]))
)


test_cleaned['Leather interior'] = (
    test_cleaned.groupby('Model')['Leather interior']
     .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else test_cleaned['Leather interior'].mode().iloc[0]))
)


train_cleaned['Levy'] = (
    train_cleaned.groupby('Model')['Levy']
     .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else train_cleaned['Levy'].mode().iloc[0]))
)


test_cleaned['Levy'] = (
    test_cleaned.groupby('Model')['Levy']
     .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else test_cleaned['Levy'].mode().iloc[0]))
)


train_cleaned['Doors'] = (
    train_cleaned.groupby('Model')['Doors']
     .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else train_cleaned['Doors'].mode().iloc[0]))
)


test_cleaned['Doors'] = (
    test_cleaned.groupby('Model')['Doors']
     .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else test_cleaned['Doors'].mode().iloc[0]))
)


train_cleaned['Drive wheels'] = (
    train_cleaned.groupby('Manufacturer')['Drive wheels']
     .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else train_cleaned['Drive wheels'].mode().iloc[0]))
)


test_cleaned['Drive wheels'] = (
    test_cleaned.groupby('Manufacturer')['Drive wheels']
     .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else test_cleaned['Drive wheels'].mode().iloc[0]))
)


train_cleaned['Wheel'] = (
    train_cleaned.groupby('Manufacturer')['Wheel']
     .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else train_cleaned['Wheel'].mode().iloc[0]))
)


test_cleaned['Wheel'] = (
    test_cleaned.groupby('Manufacturer')['Wheel']
     .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else test_cleaned['Wheel'].mode().iloc[0]))
)


train_cleaned['Cylinders'] = (
    train_cleaned.groupby('Engine volume')['Cylinders']
    .transform(lambda x: x.fillna(
        x.mode().iloc[0] if not x.mode().empty
        else train_cleaned['Cylinders'].mode().iloc[0]
    ))
)


test_cleaned['Cylinders'] = (
    test_cleaned.groupby('Engine volume')['Cylinders']
    .transform(lambda x: x.fillna(
        x.mode().iloc[0] if not x.mode().empty
        else test_cleaned['Cylinders'].mode().iloc[0]
    ))
)


train_cleaned['Category'] = (
    train_cleaned.groupby('Model')['Category']
    .transform(lambda x: x.fillna(
        x.mode().iloc[0] if not x.mode().empty
        else train_cleaned['Category'].mode().iloc[0]
    ))
)


test_cleaned['Category'] = (
    test_cleaned.groupby('Model')['Category']
    .transform(lambda x: x.fillna(
        x.mode().iloc[0] if not x.mode().empty
        else test_cleaned['Category'].mode().iloc[0]
    ))
)


train_cleaned['Airbags'] = (
    train_cleaned.groupby('Category')['Airbags']
    .transform(lambda x: x.fillna(
        x.mode().iloc[0] if not x.mode().empty
        else train_cleaned['Airbags'].mode().iloc[0]
    ))
)


test_cleaned['Airbags'] = (
    test_cleaned.groupby('Category')['Airbags']
    .transform(lambda x: x.fillna(
        x.mode().iloc[0] if not x.mode().empty
        else test_cleaned['Airbags'].mode().iloc[0]
    ))
)


train_cleaned['Color'] = train_cleaned['Color'].fillna(train_cleaned['Color'].mode().iloc[0])


test_cleaned['Color'] = test_cleaned['Color'].fillna(test_cleaned['Color'].mode().iloc[0])


train_cleaned['Price'] = (
    train_cleaned.groupby(['Model','Prod. year'])['Price']
    .transform(lambda x: x.fillna(
        x.mode().iloc[0] if not x.mode().empty
        else train_cleaned['Price'].mode().iloc[0]
    ))
)


test_cleaned['Price'] = (
    test_cleaned.groupby(['Model','Prod. year'])['Price']
    .transform(lambda x: x.fillna(
        x.mode().iloc[0] if not x.mode().empty
        else test_cleaned['Price'].mode().iloc[0]
    ))
)


numeric_columns = ['ID', 'Price', 'Prod. year', 'Airbags']
for col in numeric_columns:
    train_cleaned[col] = pd.to_numeric(train_cleaned[col], errors='coerce')

train_cleaned['Levy'] = pd.to_numeric(train_cleaned['Levy'], errors='coerce')
train_cleaned['Levy'].fillna(train_cleaned['Levy'].mean(), inplace=True)

train_cleaned['Mileage'] = train_cleaned['Mileage'].astype(str).str.replace(' km', '').str.replace(',', '').astype(float)

train_cleaned['Engine volume'] = train_cleaned['Engine volume'].str.extract(r'([\d.]+)').astype(float)

train_cleaned['Cylinders'] = pd.to_numeric(train_cleaned['Cylinders'], errors='coerce')

categorical_columns = [
    'Manufacturer', 'Model', 'Category', 'Leather interior', 'Fuel type',
    'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color'
]
for col in categorical_columns:
    train_cleaned[col] = train_cleaned[col].fillna('Unknown')

for col in train_cleaned.columns:
    print(f"\nColumn: {col}")
    print(train_cleaned[col].unique())

train_cleaned = train_cleaned[train_cleaned['Manufacturer'] != 'Ã©Â\x86Â¿Ã¢Â\x99Â\x82Ã¥Â\x84ÂºÃ©Â\x86Â¿Ã¦Â\x9bÂ\x96Ã¥Â\x84Â\x9b']


train_cleaned = train_cleaned.dropna(subset=['Prod. year'])


train_cleaned = train_cleaned[~train_cleaned['Category'].isin(['Yes', 'No'])]


train_cleaned = train_cleaned[train_cleaned['Leather interior'].isin(['Yes', 'No'])]


valid_fuels = ['Petrol', 'Diesel', 'LPG', 'Hybrid', 'CNG', 'Plug-in Hybrid', 'Hydrogen']
train_cleaned = train_cleaned[train_cleaned['Fuel type'].isin(valid_fuels)]


min_valid = 0.5
max_valid = 10

valid_values = train_cleaned['Engine volume'][(train_cleaned['Engine volume'] >= min_valid) & (train_cleaned['Engine volume'] <= max_valid)]
mean_value = valid_values.mean()

train_cleaned['Engine volume'] = train_cleaned['Engine volume'].apply(lambda x: mean_value if x < min_valid or x > max_valid else x)

print(train_cleaned['Engine volume'].unique())

train_cleaned = train_cleaned.dropna(subset=['Cylinders'])


valid_gearbox_types = ['Automatic', 'Tiptronic', 'Manual', 'Variator']
train_cleaned = train_cleaned[train_cleaned['Gear box type'].isin(valid_gearbox_types)]


valid_drive_wheel_types = ['Front', '4x4', 'Rear']
train_cleaned = train_cleaned[train_cleaned['Drive wheels'].isin(valid_drive_wheel_types)]


door_corrections = {
    '4-May': '4',
    '2-Mar': '2',
    '>5': '>5'
}
train_cleaned['Doors'] = train_cleaned['Doors'].replace(door_corrections)


valid_wheel_values = ['Left wheel', 'Right-hand drive']

wheel_mode = train_cleaned[train_cleaned['Wheel'].isin(valid_wheel_values)]['Wheel'].mode()[0]

train_cleaned['Wheel'] = train_cleaned['Wheel'].apply(
    lambda x: x if x in valid_wheel_values else wheel_mode
)


numeric_columns = ['ID', 'Price', 'Prod. year', 'Airbags']
for col in numeric_columns:
    test_cleaned[col] = pd.to_numeric(train_cleaned[col], errors='coerce')

test_cleaned['Levy'] = pd.to_numeric(train_cleaned['Levy'], errors='coerce')
test_cleaned['Levy'].fillna(train_cleaned['Levy'].mean(), inplace=True)

test_cleaned['Mileage'] = test_cleaned['Mileage'].astype(str).str.replace(' km', '').str.replace(',', '').astype(float)

test_cleaned['Engine volume'] = test_cleaned['Engine volume'].str.extract(r'([\d.]+)').astype(float)

test_cleaned['Cylinders'] = pd.to_numeric(train_cleaned['Cylinders'], errors='coerce')

categorical_columns = [
    'Manufacturer', 'Model', 'Category', 'Leather interior', 'Fuel type',
    'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color'
]
for col in categorical_columns:
    test_cleaned[col] = test_cleaned[col].fillna('Unknown')

for col in test_cleaned.columns:
    print(f"\nColumn: {col}")
    print(test_cleaned[col].unique())


test_cleaned['Price'].fillna(test_cleaned['Price'].median(), inplace=True)


def clean_model_name(name):
    return re.sub(r'[^\x00-\x7F]+', '', name)

test_cleaned['Model'] = test_cleaned['Model'].apply(clean_model_name)


test_cleaned['Prod. year'] = test_cleaned['Prod. year'].fillna(test_cleaned['Prod. year'].mean())
test_cleaned['Prod. year'] = test_cleaned['Prod. year'].astype(int)


mode_value = test_cleaned['Category'].mode()[0]
test_cleaned['Category'] = test_cleaned['Category'].replace({'No': mode_value, 'Yes': mode_value})

print(test_cleaned['Category'].unique())

mode_value = test_cleaned['Leather interior'][test_cleaned['Leather interior'].isin(['Yes', 'No'])].mode()[0]
test_cleaned['Leather interior'] = test_cleaned['Leather interior'].apply(lambda x: mode_value if x not in ['Yes', 'No'] else x)


valid_fuel_types = ['Diesel', 'Hybrid', 'Petrol', 'LPG', 'CNG', 'Plug-in Hybrid']
mode_value = test_cleaned['Fuel type'][test_cleaned['Fuel type'].isin(valid_fuel_types)].mode()[0]

test_cleaned['Fuel type'] = test_cleaned['Fuel type'].apply(lambda x: mode_value if x not in valid_fuel_types else x)


min_valid = 0.5
max_valid = 10

valid_values = test_cleaned['Engine volume'][(test_cleaned['Engine volume'] >= min_valid) & (test_cleaned['Engine volume'] <= max_valid)]
mean_value = valid_values.mean()

test_cleaned['Engine volume'] = test_cleaned['Engine volume'].apply(lambda x: mean_value if x < min_valid or x > max_valid else x)


test_cleaned['Cylinders'].fillna(test_cleaned['Cylinders'].mode()[0], inplace=True)


test_cleaned['Gear box type'] = test_cleaned['Gear box type'].replace([np.nan, 'Front', 'Rear', '4x4'], test_cleaned['Gear box type'].mode()[0])


test_cleaned['Drive wheels'] = test_cleaned['Drive wheels'].replace(['2-Mar', '4-May'], test_cleaned['Drive wheels'].mode()[0])


door_corrections = {
    '4-May': '4',
    '2-Mar': '2',
    '>5': '>5'
}
test_cleaned['Doors'] = test_cleaned['Doors'].replace(door_corrections)
test_cleaned['Doors'] = test_cleaned['Doors'].replace(['Left wheel'], test_cleaned['Doors'].mode()[0])


valid_fuel_types = ['Left wheel', 'Right-hand drive']
mode_value = test_cleaned['Wheel'][test_cleaned['Wheel'].isin(valid_fuel_types)].mode()[0]

test_cleaned['Wheel'] = test_cleaned['Wheel'].apply(lambda x: mode_value if x not in valid_fuel_types else x)


valid_colors = ['Grey', 'Black', 'Silver', 'Red', 'White', 'Blue', 'Brown', 'Carnelian red',
                'Beige', 'Orange', 'Green', 'Golden', 'Sky blue', 'Purple', 'Yellow', 'Pink']

mode_value = test_cleaned['Color'][test_cleaned['Color'].isin(valid_colors)].mode()[0]

test_cleaned['Color'] = test_cleaned['Color'].apply(lambda x: mode_value if x not in valid_colors else x)


test_cleaned['Airbags'].fillna(test_cleaned['Airbags'].mode()[0], inplace=True)


for col in train_cleaned.columns:
    print(f"\nColumn: {col}")
    print(train_cleaned[col].unique())

print(train_cleaned['Price'].describe())


Q1 = train_cleaned['Price'].quantile(0.25)
Q3 = train_cleaned['Price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

price_median = train_cleaned['Price'].median()

train_cleaned['Price'] = train_cleaned['Price'].apply(
    lambda x: price_median if x < lower_bound or x > upper_bound else x
)




Q1 = train_cleaned['Levy'].quantile(0.25)
Q3 = train_cleaned['Levy'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

median_levy = train_cleaned['Levy'].median()
train_cleaned['Levy'] = train_cleaned['Levy'].apply(lambda x: median_levy if x < lower_bound or x > upper_bound else x)


print(train_cleaned['Prod. year'].describe())


valid_years = (train_cleaned['Prod. year'] >= 1980) & (train_cleaned['Prod. year'] <= 2025)

median_prod_year = train_cleaned.loc[valid_years, 'Prod. year'].median()

train_cleaned.loc[~valid_years, 'Prod. year'] = median_prod_year


threshold_mileage = 300000

train_cleaned.loc[train_cleaned['Mileage'] > threshold_mileage, 'Mileage'] = train_cleaned['Mileage'].median()


valid_cylinders = [4, 6, 8, 12]
mode_cylinders = train_cleaned['Cylinders'].mode()[0]

train_cleaned.loc[~train_cleaned['Cylinders'].isin(valid_cylinders), 'Cylinders'] = mode_cylinders


for col in test_cleaned.columns:
    print(f"\nColumn: {col}")
    print(test_cleaned[col].unique())


Q1 = test_cleaned['Price'].quantile(0.25)
Q3 = test_cleaned['Price'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = (test_cleaned['Price'] < lower_bound) | (test_cleaned['Price'] > upper_bound)

mean_value = test_cleaned['Price'].mean()

test_cleaned['Price'] = np.where(outliers, mean_value, test_cleaned['Price'])


Q1 = test_cleaned['Levy'].quantile(0.25)
Q3 = test_cleaned['Levy'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = (test_cleaned['Levy'] < lower_bound) | (test_cleaned['Levy'] > upper_bound)

mean_without_outliers = test_cleaned.loc[~outliers, 'Levy'].mean()
test_cleaned['Levy'] = np.where(outliers, mean_without_outliers, test_cleaned['Levy'])


Q1_year = test_cleaned['Prod. year'].quantile(0.25)
Q3_year = test_cleaned['Prod. year'].quantile(0.75)
IQR_year = Q3_year - Q1_year
lower_bound_year = Q1_year - 1.5 * IQR_year
upper_bound_year = Q3_year + 1.5 * IQR_year

outliers_year = (test_cleaned['Prod. year'] < lower_bound_year) | (test_cleaned['Prod. year'] > upper_bound_year)

mean_year = round(test_cleaned.loc[~outliers_year, 'Prod. year'].mean())
test_cleaned['Prod. year'] = np.where(outliers_year, mean_year, test_cleaned['Prod. year'])


Q1_mileage = test_cleaned['Mileage'].quantile(0.25)
Q3_mileage = test_cleaned['Mileage'].quantile(0.75)
IQR_mileage = Q3_mileage - Q1_mileage
lower_bound_mileage = Q1_mileage - 1.5 * IQR_mileage
upper_bound_mileage = Q3_mileage + 1.5 * IQR_mileage

outliers_mileage = (test_cleaned['Mileage'] < lower_bound_mileage) | (test_cleaned['Mileage'] > upper_bound_mileage)
mean_mileage = test_cleaned.loc[~outliers_mileage, 'Mileage'].mean()
test_cleaned['Mileage'] = np.where(outliers_mileage, mean_mileage, test_cleaned['Mileage'])


train_cleaned.info()

from datetime import datetime
current_year = datetime.now().year

train_cleaned['car_age'] = current_year - train_cleaned['Prod. year']

train_cleaned.head()

sns.histplot(train_cleaned['Price'], kde=True)

plot_data = train_cleaned.copy(deep=True)

plot_data.info()



temp_data = plot_data.copy()

temp_data = temp_data.dropna(subset=["Prod. year", "Price"])

avg_price_per_year = temp_data.groupby("Prod. year")["Price"].mean().reset_index()

avg_price_per_airbag = plot_data.groupby('Airbags')['Price'].mean().reset_index()


avg_price_per_levy = plot_data.groupby('Levy')['Price'].mean().reset_index()

train_cleaned.to_csv('not_encoded_train.csv', index=False)
test_cleaned.to_csv('not_encoded_test.csv', index=False)

################# encoding
M_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

M_encoder.fit(train_cleaned[['Manufacturer']])
joblib.dump(M_encoder, 'encoders/manufacturer_encoder_1.pkl')

train_cleaned['Manufacturer'] = M_encoder.transform(train_cleaned[['Manufacturer']])
test_cleaned['Manufacturer'] = M_encoder.transform(test_cleaned[['Manufacturer']])


C_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on train only
C_encoder.fit(train_cleaned[['Category']])
joblib.dump(C_encoder, 'encoders/category_encoder_1.pkl')

# Transform both
train_cleaned['Category'] = C_encoder.transform(train_cleaned[['Category']])
test_cleaned['Category'] = C_encoder.transform(test_cleaned[['Category']])


L_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on train only
L_encoder.fit(train_cleaned[['Leather interior']])
joblib.dump(L_encoder, 'encoders/leather_encoder_1.pkl')

# Transform both
train_cleaned['Leather interior'] = L_encoder.transform(train_cleaned[['Leather interior']])
test_cleaned['Leather interior'] = L_encoder.transform(test_cleaned[['Leather interior']])


F_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on train only
F_encoder.fit(train_cleaned[['Fuel type']])
joblib.dump(F_encoder, 'encoders/fuel_encoder_1.pkl')
# Transform both
train_cleaned['Fuel type'] = F_encoder.transform(train_cleaned[['Fuel type']])
test_cleaned['Fuel type'] = F_encoder.transform(test_cleaned[['Fuel type']])


Model_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on train only
Model_encoder.fit(train_cleaned[['Model']])
joblib.dump(Model_encoder, 'encoders/model_encoder_1.pkl')

# Transform both
train_cleaned['Model'] = Model_encoder.transform(train_cleaned[['Model']])
test_cleaned['Model'] = Model_encoder.transform(test_cleaned[['Model']])


D_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on train only
D_encoder.fit(train_cleaned[['Drive wheels']])
joblib.dump(D_encoder, 'encoders/drive_encoder.pkl')

# Transform both
train_cleaned['Drive wheels'] = D_encoder.transform(train_cleaned[['Drive wheels']])
test_cleaned['Drive wheels'] = D_encoder.transform(test_cleaned[['Drive wheels']])


W_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on train only
W_encoder.fit(train_cleaned[['Wheel']])
joblib.dump(W_encoder, 'encoders/wheel_encoder_1.pkl')

# Transform both
train_cleaned['Wheel'] = W_encoder.transform(train_cleaned[['Wheel']])
test_cleaned['Wheel'] = W_encoder.transform(test_cleaned[['Wheel']])


Co_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on train only
Co_encoder.fit(train_cleaned[['Color']])
joblib.dump(Co_encoder, 'encoders/color_encoder_1.pkl')

# Transform both
train_cleaned['Color'] = Co_encoder.transform(train_cleaned[['Color']])
test_cleaned['Color'] = Co_encoder.transform(test_cleaned[['Color']])


Door_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on train only
Door_encoder.fit(train_cleaned[['Doors']])
joblib.dump(Door_encoder, 'encoders/doors_encoder_1.pkl')

# Transform both
train_cleaned['Doors'] = Door_encoder.transform(train_cleaned[['Doors']])
test_cleaned['Doors'] = Door_encoder.transform(test_cleaned[['Doors']])


Gear_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on train only
Gear_encoder.fit(train_cleaned[['Gear box type']])
joblib.dump(Gear_encoder, 'encoders/gear_encoder_1.pkl')

# Transform both
train_cleaned['Gear box type'] = Gear_encoder.transform(train_cleaned[['Gear box type']])
test_cleaned['Gear box type'] = Gear_encoder.transform(test_cleaned[['Gear box type']])

train_cleaned = train_cleaned.drop('ID', axis=1)

#from sklearn.preprocessing import StandardScaler

#numeric_cols = ['Levy', 'Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags']
#scaler = StandardScaler()
#train_cleaned[numeric_cols] = scaler.fit_transform(train_cleaned[numeric_cols])
#test_cleaned[numeric_cols] = scaler.transform(test_cleaned[numeric_cols])

"""# checking correlation"""

# Calculate the correlation matrix

## Select only numerical columns
numerical_columns = train_cleaned.select_dtypes(include=np.number).columns.drop("Price")
correlation_matrix = train_cleaned[numerical_columns].corr()


# Create a heatmap of the correlation matrix


# Scatter plot: Production Year vs Price

"""# feature selection"""

# Calculate the correlation between numerical features and the label
correlations_with_target = train_cleaned[numerical_columns].corrwith(train_cleaned['Price'])


sorted_features = correlations_with_target.abs().sort_values(ascending=False)

train_cleaned.to_csv('train_cleaned_new_1.csv', index=False)
test_cleaned.to_csv('test_cleaned_new_1.csv', index=False)

"""# linear regression model"""


# Select features and target
selected_features = ['Airbags', 'Prod. year', 'Mileage','Drive wheels','Wheel',
                     'Gear box type', 'Manufacturer','Model','Levy','Fuel type','Leather interior','Category','Cylinders']

