**Credit Card Fraud Detection**

This project aims to detect fraudulent credit card transactions using a dataset containing various transaction and user attributes. The model uses Logistic Regression to predict whether a transaction is fraudulent. The project includes data preprocessing, feature engineering, model training, evaluation, and visualization.

**Dataset**

The dataset is collected from:-

https://www.kaggle.com/datasets/kartik2112/fraud-detection




**The dataset used in this project includes training and test sets with the following columns:**


Unnamed: 0: Row index.

trans_date_trans_time: Transaction date and time.

cc_num: Credit card number.

merchant: Merchant name.

category: Merchant category.

amt: Transaction amount.

first: Cardholder's first name.

last: Cardholder's last name.

gender: Cardholder's gender.

street: Cardholder's street.

city: Cardholder's city.

state: Cardholder's state.

zip: Cardholder's zip code.

lat: Cardholder's latitude.

long: Cardholder's longitude.

city_pop: Population of the cardholder's city.

job: Cardholder's job.

dob: Cardholder's date of birth.

trans_num: Transaction number.

merchant_lat: Merchant's latitude.

merchant_long: Merchant's longitude.

is_fraud: Whether the transaction is fraudulent (1) or not (0).


**Project Structure**

data_loading: Load the dataset and handle missing values.

data_preprocessing: Convert date columns, drop irrelevant columns, and encode categorical variables.

data_visualization: Visualize distributions of categorical variables, transaction times, and fraud class distribution.

feature_engineering: Create new features from existing data.

model_training: Split the data, initialize and train a Logistic Regression model.

model_evaluation: Evaluate the model using accuracy, precision, recall, F1 score, and ROC AUC score.

**Installation**

To run this project, ensure you have the following libraries installed:

numpy
pandas
matplotlib
seaborn
scikit-learn

You can install these libraries using pip:
pip install<'library name'>




**Results**

The model's performance is evaluated based on accuracy, precision, recall, F1 score, and ROC AUC score.
These metrics provide a comprehensive understanding of how well the model detects fraudulent transactions.

**License**

This project is licensed under the MIT License. See the LICENSE file for more details.
