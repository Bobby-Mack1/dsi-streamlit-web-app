#import libraries
import streamlit as st
import pandas as pd
import joblib


#load pipeline
model=joblib.load('model.joblib')

#add title and instructions
st.title('Purchase Prediction Model')
st.subheader('enter customer infor and submit for likelihood to purchase')

#age input form
age=st.number_input(
    label='01. enter the customers age',
    min_value=18,
    max_value=120,
    value=35)

#gender input form
gender=st.radio(
    label='02. enter customers gender',
    options=['M','F'])

#credit score input form
credit_score=st.number_input(
    label='03. enter the customers credit score',
    min_value=0,
    max_value=1000,
    value=500)

#submit inputs to model 
if st.button('submit for prediction'):
    #store data in a df fro prediction
    new_data= pd.DataFrame({'age' :[age], 'gender':[gender], 'credit_score':[credit_score]})
    #apply model pipeline to input data and extract prob prediction
    pred_proba=model.predict_proba(new_data)[0][1]
    #output prediction
    st.subheader(f'Based on these attributes our model predicts a purchase probability of {pred_proba:.0%}')

















