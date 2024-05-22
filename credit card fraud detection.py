import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Locate files within input directory
input_dir = '/kaggle/input'
for root_dir, _, files in os.walk(input_dir):
    for file_name in files:
        print(os.path.join(root_dir, file_name))

# Read the training dataset
train_dataset_path = 'C:\\Users\\SURYA VARMA\\OneDrive\\Desktop\\Credit Card Fraud Detection\\datasets\\fraudTrain.csv'
data=pd.read_csv(train_dataset_path)
# Output dataset overview
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Plot distributions of categorical variables
cat_features = ['merchant', 'category', 'gender', 'city', 'job']
for feature in cat_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=feature, data=data, palette='Set2')
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45)
    plt.show()

# Convert date columns to datetime
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['dob'] = pd.to_datetime(data['dob'])

# Analyze transaction times
plt.figure(figsize=(12, 6))
data['trans_date_trans_time'].dt.hour.plot(kind='hist', bins=24, color='skyblue', edgecolor='black')
plt.title('Hourly Transaction Distribution')
plt.xlabel('Hour')
plt.ylabel('Frequency')
plt.show()

# Visualize fraud class distribution
plt.figure(figsize=(7, 5))
data['is_fraud'].value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Geospatial plot of transactions
plt.figure(figsize=(12, 10))
plt.scatter(data['merch_long'], data['merch_lat'], c=data['is_fraud'], cmap='coolwarm', alpha=0.6)
plt.title('Transaction Geospatial Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Fraud Indicator')
plt.show()

# Feature Engineering
data['amt_decimal'] = data['amt'] % 1
data['transaction_date'] = data['trans_date_trans_time'].dt.date
data['cardholder_age'] = (pd.to_datetime(data['transaction_date']) - data['dob']).dt.days // 365
data['amt_to_city_pop_ratio'] = data['amt'] / data['city_pop']

# Preview new features
print(data[['amt_decimal', 'cardholder_age', 'amt_to_city_pop_ratio']].head())

# Drop irrelevant columns
drop_columns = ['Unnamed: 0', 'cc_num', 'trans_date_trans_time', 'transaction_date', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num']
data.drop(columns=[col for col in drop_columns if col in data.columns], inplace=True)
print(data.head())

# Load and preprocess test data
test_dataset_path = 'C:\\Users\\SURYA VARMA\\OneDrive\\Desktop\\\Credit Card Fraud Detection\\datasets\\fraudTest.csv'
test_data = pd.read_csv(test_dataset_path)
test_data.dropna(inplace=True)

# Apply same feature engineering to test data
test_data['amt_decimal'] = test_data['amt'] % 1
test_data['trans_date_trans_time'] = test_data['trans_date_trans_time'].astype(str)
test_data['transaction_date'] = pd.to_datetime(test_data['trans_date_trans_time'].str.split(' ').str[0])
test_data['cardholder_age'] = (test_data['transaction_date'] - pd.to_datetime(test_data['dob'])).dt.days // 365
test_data['amt_to_city_pop_ratio'] = test_data['amt'] / test_data['city_pop']
print(test_data[['amt_decimal', 'cardholder_age', 'amt_to_city_pop_ratio']].head())

# Drop irrelevant columns from test data
test_data.drop(columns=[col for col in drop_columns if col in test_data.columns], inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
for col in cat_features:
    if col in data.columns:
        data[col] = encoder.fit_transform(data[col].astype(str))
    if col in test_data.columns:
        test_data[col] = encoder.fit_transform(test_data[col].astype(str))

# Prepare training and validation data
X_train = data.drop(columns=['is_fraud'])
y_train = data['is_fraud']
X_val = test_data.drop(columns=['is_fraud'])
y_val = test_data['is_fraud']

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred)

# Print evaluation metrics
print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")