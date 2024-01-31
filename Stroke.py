import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import math
from pandas.api.types import is_numeric_dtype

df = pd.read_csv('C:/Users/{insert path}/healthcare-dataset-stroke-data.csv')
print(df)

print(df.isna())
print(df.describe())
print(df[df["bmi"].isnull()])
df = df.dropna()
print(df[df["bmi"].isnull()])
df = df.drop(columns=["id"])
print(df)
df = df.drop(df[df['gender'] == 'Other'].index)
#Logistic Regression
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1}).astype(float)

df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1}).astype(float)
df['work_type'] = df['work_type'].map({'Self-employed': 0, 'Private': 1, 'children':2, 'Govt_job':3}).astype(float)
df['Residence_type'] = df['Residence_type'].map({'Rural': 0, 'Urban': 1}).astype(float)
df['smoking_status'] = df['smoking_status'].map({'never smoked': 0, 'formerly smoked': 1, 'Unknown':2}).astype(float)
feature_cols = ['gender', 'age', 'heart_disease', 'ever_married','avg_glucose_level', 'bmi']
X = df[feature_cols]
print(X)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(random_state=42)
(model.fit(X_train, y_train))
y_pred = model.predict(X_test)
print(y_pred)
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
score = model.score(X_test, y_test)
print(score)
print(classification_report(y_test, y_pred))

#uSE smoTE to oversample the strokes in logistic regression
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_smote, y_train_smote)   
y_pred = logistic_regression.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
print(model.score(X_test, y_pred))
print(model.coef_, model.intercept_)

beta = model.coef_
beta_intercept =(model.intercept_)
beta_coef = pd.DataFrame(beta, columns=['gender', 'age', 'heart_disease', 'ever_married','avg_glucose_level', 'bmi'])
print(beta_coef)
beta_int = pd.DataFrame(beta_intercept, columns=['beta_0'])
print(beta_coef)
print(beta_int)

print(is_numeric_dtype(beta))
print(is_numeric_dtype(beta_coef))
print(beta_coef.dtypes)
'''
gender = input("What is your gender? ")
age = float(input("How old are you? "))
heart_disease = input("Have you ever had heart disease? ")
married = input("Have you ever been/are married? ")
glucose = float(input("What is your glucose level? "))
bmi = float(input("What is your Body Mass Index (BMI)? "))

user_input = pd.DataFrame(data=[[gender,age,heart_disease,married,glucose,bmi]], columns=['gender', 'age', 'heart_disease', 'ever_married','avg_glucose_level', 'bmi'])


user_input['gender'] = user_input['gender'].map({'Male': 0, 'male':0, 'Female': 1, 'female':1}).astype(float)
user_input['ever_married'] = user_input['ever_married'].map({'No': 0, 'no':0, 'Yes': 1, 'yes':1}).astype(float)
user_input['heart_disease'] = user_input['heart_disease'].map({'No': 0, 'no':0, 'Yes': 1, 'yes':1}).astype(float)
print(user_input)
user_predict = model.predict(user_input)
print(user_predict)
print(user_input.dtypes)
print(user_input)
print(beta_coef)

user_prediction = user_input.mul(beta_coef)
print(user_prediction)
user_prediction["beta_0"] = beta_int
user_prediction["Sum"] = user_prediction[list(user_prediction.columns)].sum(axis=1)
print(user_prediction)

user_prediction_sum = user_prediction["Sum"]
user_prediction_sum = user_prediction_sum.to_numpy()
print(user_prediction_sum)
user_prediction_sum = np.exp(user_prediction_sum)
print(user_prediction_sum)
prob = (user_prediction_sum) / (1+(user_prediction_sum))
print(f"The probability you wil have a stroke right now is {prob}")
'''
#See how each variable is correlated with Strokes
print(df.corr())

#Maybe add some other columns like smoking and residence and jobs to make model more accurate\
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('C:/Users/{insert path}/healthcare-dataset-stroke-data.csv')
print(df)
df = df.dropna()
print(df[df["bmi"].isnull()])
df = df.drop(columns=["id"])
print(df)
df = df.drop(df[df['gender'] == 'Other'].index)
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1}).astype(float)

df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1}).astype(float)
df['work_type'] = df['work_type'].map({'Self-employed': 0, 'Private': 1, 'children':2, 'Govt_job':3}).astype(float)
df['Residence_type'] = df['Residence_type'].map({'Rural': 0, 'Urban': 1}).astype(float)
df['smoking_status'] = df['smoking_status'].map({'never smoked': 0, 'smokes': 1, 'formerly smoked':2, 'Unknown':3}).astype(float)

print(df["smoking_status"])
feature_cols = ['gender', 'age', 'heart_disease','avg_glucose_level', 'bmi', 'smoking_status', 'hypertension']
X = df[feature_cols]
print(X)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(random_state=42, max_iter=1000)
(model.fit(X_train, y_train))
y_pred = model.predict(X_test)
print(y_pred)

cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
score = model.score(X_test, y_test)
print(score)
print(classification_report(y_test, y_pred))
smote = SMOTE(random_state=0)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_smote, y_train_smote)   
y_pred = logistic_regression.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
print(model.score(X_test, y_pred))
print(model.coef_, model.intercept_)

beta = model.coef_
beta_intercept =(model.intercept_)
beta_coef = pd.DataFrame(beta, columns=['gender', 'age', 'heart_disease','avg_glucose_level', 'bmi', 'smoking_status', 'hypertension'])
print(beta_coef)
beta_int = pd.DataFrame(beta_intercept, columns=['beta_0'])
print(beta_coef)
print(beta_int)

print(is_numeric_dtype(beta))
print(is_numeric_dtype(beta_coef))
print(beta_coef.dtypes)


gender = input("What is your gender? ")
age = float(input("How old are you? "))
heart_disease = input("Have you ever had heart disease? ")
glucose = float(input("What is your glucose level? "))
bmi = float(input("What is your Body Mass Index (BMI)? "))
smoking_status = input("Do you smoke? (Insert formerly smoked if you have quit smoking)? ")
hypertension = input("Do you have hypertension? ")

user_input = pd.DataFrame(data=[[gender,age,heart_disease,glucose,bmi, smoking_status, hypertension]], columns=['gender', 'age', 'heart_disease','avg_glucose_level', 'bmi', 'smoking_status', 'hypertension'])


user_input['gender'] = user_input['gender'].map({'Male': 0, 'male':0, 'Female': 1, 'female':1}).astype(float)
user_input['heart_disease'] = user_input['heart_disease'].map({'No': 0, 'no':0, 'Yes': 1, 'yes':1}).astype(float)
user_input['smoking_status'] = user_input['smoking_status'].map({'No': 0, 'no':0, 'Yes': 1, 'yes':1, 'formerly smoked':2}).astype(float)
user_input['hypertension'] = user_input['hypertension'].map({'No': 0, 'no':0, 'Yes': 1, 'yes':1}).astype(float)

user_predict = model.predict(user_input)
print(user_predict)
print(user_input.dtypes)
print(user_input)

user_prediction = user_input.mul(beta_coef)
print(user_prediction)
user_prediction["beta_0"] = beta_int
user_prediction["Sum"] = user_prediction[list(user_prediction.columns)].sum(axis=1)
print(user_prediction)

user_prediction_sum = user_prediction["Sum"]
user_prediction_sum = user_prediction_sum.to_numpy()
print(user_prediction_sum)
user_prediction_sum = np.exp(user_prediction_sum)
print(user_prediction_sum)
prob = (user_prediction_sum) / (1+(user_prediction_sum))
print(f"The probability you wil have a stroke is {prob}")
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test ,y_pred)
print(f"The accuracy of this model is: {score}")
