import streamlit as st
import pandas as pd
import pickle as pk
import os


BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR,"load_Prediction_Web", "model.pkl")
scaler_path = os.path.join(BASE_DIR,"load_Prediction_Web", "scaler.pkl")

model = pk.load(open(model_path, "rb"))
scaler = pk.load(open(scaler_path, "rb"))


st.header('Loan Prediction App')

no_of_dep = st.slider('Choose No of dependents', 0, 5)
grad = st.selectbox('Choose Education', ['Graduated', 'Not Graduated'])
self_emp = st.selectbox('Self Emoployed ?', ['Yes', 'No'])
Annual_Income = st.slider('Choose Annual Income', 0, 10000000)
Loan_Amount = st.slider('Choose Loan Amount', 0, 10000000)
Loan_Dur = st.slider('Choose Loan Duration', 0, 20)
Cibil = st.slider('Choose Cibil Score', 0, 1000)
Assets = st.slider('Choose Assets', 0, 10000000)

grad_s = 0 if grad == 'Graduated' else 1
emp_s = 0 if self_emp == 'No' else 1

if st.button("Predict"):
    pred_data = pd.DataFrame(
        [[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
        columns=[
            'no_of_dependents',
            'education',
            'self_employed',
            'income_annum',
            'loan_amount',
            'loan_term',
            'cibil_score',
            'Assets'
        ]
    )

    pred_data = scaler.transform(pred_data)
    prediction = model.predict(pred_data)

    if prediction[0] == 1:
        st.success('Loan Is Approved ✅')
    else:
        st.error('Loan Is Rejected ❌')
