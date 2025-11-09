import joblib 
import pandas as pd
import numpy as np
import tensorflow as tf
preprocessor=joblib.load('preprocessor.pkl')
model=tf.keras.models.load_model('churn_model.h5')
def churn_prediction(dataset):
    data_preprocessed=preprocessor.transform(dataset)
    prediction=model.predict(data_preprocessed)
    if prediction>=0.5:
        prediction=1
    else:
        prediction=0
    return prediction

at_risk_customer = {
    'CreditScore': 608,
    'Geography': 'Germany', # High churn group
    'Gender': 'Female',
    'Age': 41,
    'Tenure': 1,
    'Balance': 125510.82,    # Customers with a balance are more likely to churn
    'NumOfProducts': 1,      # Customers with 1 product are more likely to churn
    'HasCrCard': 1,
    'IsActiveMember': 0,     # Inactive members are more likely to churn
    'EstimatedSalary': 19000.0
}

loyal_customer = {
    'CreditScore': 750,
    'Geography': 'France',   # Low churn group
    'Gender': 'Male',
    'Age': 30,
    'Tenure': 5,
    'Balance': 0.0,          # Customers with 0 balance are more likely to stay
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,     # Active members are more likely to stay
    'EstimatedSalary': 150000.0
}

df_1=pd.DataFrame([at_risk_customer],columns=at_risk_customer.keys())
df_2=pd.DataFrame([loyal_customer],columns=loyal_customer.keys())

print(churn_prediction(df_1))  # Expected output: 1 (churn)
print(churn_prediction(df_2))  # Expected output: 0 (no churn)

# as you can see after running this code that the model is able to predict churn for both at risk and loyal customers accurately. so this project was a success!. this is my first end to end DL project. that is not based on toy data.