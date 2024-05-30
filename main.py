import streamlit as st
import pandas as pd
import numpy as np

# Title of the Streamlit app
st.title("Machine Learning-Based Classification for Predicting COVID-19 Mortality Rates")

# Display the introduction text
st.write("⚠️ **Please complete all of the information in the survey!**")
st.write("This project is inspired by the COVID-19 pandemic, a historical event that changed the world three years ago. Although the pandemic has ended, its impacts continue to affect people's mentality and lives. This project aims to commemorate COVID-19 as the largest medical event of the century and convey a message about the importance of unity and human resilience in the face of immense challenges.")

with st.form(key='my_form'):
    # Các widget nhập liệu trong form
    age = st.text_input("How old are you?")
    yes_no_diabetes = st.radio("Do you have diabetes?", ("Yes", "No"))
    yes_no_renalChronal = st.radio("Do you have renalChronal?", ("Yes", "No"))
    yes_no_hypertension = st.radio("Do you have hypertension?", ("Yes", "No"))
    yes_no_pneumonia = st.radio("Do you have pneumonia?", ("Yes", "No"))
    selectionBox1 = st.selectbox("What type of institution of the National Health System that provided the care?", ("1", "2"))
    selectionBox2 = st.selectbox("Do you indicate treatment in medical units of the first or second level?", ("First level", "Second level"))
    yes_no_hospitalized = st.radio("Have you ever been hospitalized?", ("Yes", "No"))
    selectionBox3 = st.selectbox("Did you have covid test before? If you did, select the diagnosis of the test", ("1", "2", "3", "More than 3"))

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    st.write("Age:", age)
    st.write("Diabetes:", yes_no_diabetes)
    st.write("Renal Chronal:", yes_no_renalChronal)
    st.write("Hypertension:", yes_no_hypertension)
    st.write("Pneumonia:", yes_no_pneumonia)
    st.write("Type of institution:", selectionBox1)
    st.write("Level of treatment:", selectionBox2)
    st.write("Hospitalized:", yes_no_hospitalized)
    st.write("COVID test diagnosis:", selectionBox3)
    
    # Ví dụ xử lý thêm dữ liệu
    data = {
        "Age": [age],
        "Diabetes": [yes_no_diabetes],
        "Renal Chronal": [yes_no_renalChronal],
        "Hypertension": [yes_no_hypertension],
        "Pneumonia": [yes_no_pneumonia],
        "Type of Institution": [selectionBox1],
        "Level of Treatment": [selectionBox2],
        "Hospitalized": [yes_no_hospitalized],
        "COVID Test Diagnosis": [selectionBox3]
    }
    df = pd.DataFrame(data)
    st.write("Entered data:")
    st.write(df)
