## create streamlit app to end to end deploy the model
import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

model = load_model('churn_model.h5')
scaler = pickle.load(open('standard_scaler.pkl', 'rb'))
label_encoder_gender = pickle.load(open('label_encoder.pkl', 'rb'))
one_hot_encoder_geo = pickle.load(open('one_hot_encoder.pkl', 'rb'))

## streamlit app
st.title("Customer Churn Prediction")

geography = st.selectbox("Select Geography", one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Select Gender", label_encoder_gender.classes_)
age = st.slider("Select Age", 18, 92)
balance = st.number_input("Enter Balance", min_value=0.0, max_value=250000.0)
tenure = st.slider("Select Tenure", 0, 10)
credit_score = st.number_input("Enter Credit Score", min_value=300, max_value=850)
num_of_products = st.slider("Select Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Enter Estimated Salary")

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
}

categorical_df = pd.DataFrame(
    one_hot_encoder_geo.transform([[geography]]).toarray(),
    columns=one_hot_encoder_geo.get_feature_names_out(['Geography'])
)

input_df = pd.concat([
    pd.DataFrame([input_data]).reset_index(drop=True).drop(['Geography'], axis=1),
    categorical_df
], axis=1)

input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

expected_columns = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]
input_df = input_df.reindex(columns=expected_columns, fill_value=0)

scaled_input_df = scaler.transform(input_df)
y_pred_probability = model.predict(scaled_input_df, verbose=0)[0][0]

st.write(f"Prediction probability: {y_pred_probability:.2f}")
if y_pred_probability >= 0.5:
    st.success(f"The customer is likely to churn with a probability of {y_pred_probability:.2f}")
else:
    st.info(f"The customer is not likely to churn with a probability of {y_pred_probability:.2f}")
