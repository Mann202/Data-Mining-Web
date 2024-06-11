import streamlit as st
import pandas as pd
import sklearn
import numpy as np
import pickle

# Định nghĩa trạng thái hiện tại của ứng dụng, giả sử 'main' là trang chính và 'risk_page' là trang hiện tại
current_page = st.session_state.get('page', 'risk_page')

# Load models =========================
with open('./models/knn_op.pkl', 'rb') as knn_model_file:
    knn_model = pickle.load(knn_model_file)
with open('./models/gnb_op.pkl', 'rb') as gnb_model_file:
    gnb_model = pickle.load(gnb_model_file)
with open('./models/bnb_op.pkl', 'rb') as bnb_model_file:
    bnb_model = pickle.load(bnb_model_file)
with open('./models/dt_op.pkl', 'rb') as dt_model_file:
    dt_model = pickle.load(dt_model_file)
with open('./models/knn_op.pkl', 'rb') as rf_model_file:
    rf_model = pickle.load(rf_model_file)
# =========================================

def predict_high_risk_death(model, features):
    prediction = model.predict(features)
    return prediction

# Hàm quay lại trang chính
def go_back():
    st.session_state['page'] = 'main'

if current_page == 'main':
    st.title("Machine Learning-Based Classification for Predicting COVID-19 Mortality Rates")

    st.write("⚠️ **Please complete all of the information in the survey!**")
    st.write("This project is inspired by the COVID-19 pandemic, a historical event that changed the world three years ago. Although the pandemic has ended, its impacts continue to affect people's mentality and lives. This project aims to commemorate COVID-19 as the largest medical event of the century and convey a message about the importance of unity and human resilience in the face of immense challenges.")

    with st.form(key='my_form'):
        # Các widget nhập liệu trong form
        age = st.text_input("How old are you?")

        if age:
            try:
                age = int(age)
                if age > 150 or age < 1:
                    st.warning("Tuổi phải nằm trong khoảng từ 1 đến 150.")
            except ValueError:
                st.error("Vui lòng nhập một số nguyên hợp lệ.")

        yes_no_diabetes = st.radio("Do you have diabetes?", ("Yes", "No"))
        yes_no_renalChronal = st.radio("Do you have renalChronal?", ("Yes", "No"))
        yes_no_hypertension = st.radio("Do you have hypertension?", ("Yes", "No"))
        yes_no_pneumonia = st.radio("Do you have pneumonia?", ("Yes", "No"))

        selectionBox1 = st.selectbox("What type of institution of the National Health System that provided the care?", (1,2,3,4,5,6,7,8,9,10,11,12,13))
        selectionBox2 = st.selectbox("Do you indicate treatment in medical units of the first or second level?", (1, 2))
        yes_no_hospitalized = st.radio("Have you ever been hospitalized?", ("Yes", "No"))
        selectionBox3 = st.selectbox("Did you have covid test before? If you did, select the diagnosis of the test", (1,2,3,4,5,6,7))

        st.markdown("<hr>", unsafe_allow_html=True)
        model = st.selectbox( 'Pick a model?', ('Gaussian Naives Bayes', 'Bernoulli Naives Bayes','Decicion Tree', 'Random Forest', 'K-Nearest Neighboor')) 
        
        submit_button = st.form_submit_button(label='Submit')


        if submit_button:

            yes_no_pneumonia = 1 if yes_no_pneumonia == 'Yes' else 2
            yes_no_diabetes = 1 if yes_no_diabetes == 'Yes' else 2
            yes_no_hypertension = 1 if yes_no_hypertension == 'Yes' else 2
            yes_no_renalChronal = 1 if yes_no_renalChronal == 'Yes' else 2
            yes_no_hospitalized = 1 if yes_no_hospitalized == 'Yes' else 2

            medi_unit2 = 0
            medi_unit3 = 0
            medi_unit4 = 0
            medi_unit5 = 0
            medi_unit6 = 0
            medi_unit7 = 0
            medi_unit8 = 0
            medi_unit9 = 0
            medi_unit10 = 0
            medi_unit11 = 0
            medi_unit12 = 0
            medi_unit13 = 0

            # Validate user input
            if 1 <= selectionBox1 <= 13:
                for i in range(2, 14):
                    if i == selectionBox1:
                        globals()[f"medi_unit{i}"] = 1
                        break
                    else:
                        continue

            # CLASSI_FIANL :v
            classi_final2 = 0
            classi_final3 = 0
            classi_final4 = 0
            classi_final5 = 0
            classi_final6 = 0
            classi_final7 = 0

            # Validate user input
            if 1 <= selectionBox3 <= 7:
                for i in range(2, 8):
                    if i == selectionBox3:
                        globals()[f"classi_final{i}"] = 1
                        break
                    else:
                        continue

            
            features_other = np.array([ 
            [selectionBox2, yes_no_hospitalized, yes_no_pneumonia, age, yes_no_diabetes, yes_no_hypertension, yes_no_renalChronal, medi_unit2, medi_unit3, medi_unit4, medi_unit5, medi_unit6, medi_unit7, medi_unit8, medi_unit9, medi_unit10, medi_unit11, medi_unit12, medi_unit13, classi_final2, classi_final3, classi_final4, classi_final5, classi_final6, classi_final7]], dtype= object)
            
            model = st.selectbox( 'Pick a model?', ('Gaussian Naives Bayes', 'Bernoulli Naives Bayes','Decicion Tree', 'Random Forest', 'K-Nearest Neighboor')) 
            btn = st.button("Predict")
            if btn:
                if model == 'Gaussian Naives Bayes':
                    prediction = predict_high_risk_death(gnb_model, features_other)
                elif model == 'Bernoulli Naives Bayes':
                    prediction = predict_high_risk_death(bnb_model, features_other)
                elif model == 'Decicion Tree':
                    prediction = predict_high_risk_death(dt_model, features_other)
                elif model == 'Random Forest':
                    prediction = predict_high_risk_death(rf_model, features_other)
                elif model == 'K-Nearest Neighboor':
                    prediction = predict_high_risk_death(knn_model, features_other)

            if prediction is not None: 
                if prediction[0] == 1:  
                    st.session_state['page'] = 'high-risk'
                else:
                    st.session_state['page'] = 'low-risk'
else:
    # Title của ứng dụng Streamlit
    st.title("Machine Learning-Based Classification for Predicting COVID-19 Mortality Rates")

    # Thêm CSS
    st.markdown("""
        <style>
        .reportview-container {
            background: #F7F7F9;
            padding-top: 2rem;
        }
        .custom-box {
            background-color: #EFEFEF;
            padding: 50px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .alert {
            color: black;  # Đổi màu tiêu đề thành đen
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtext {
            color: #333;
            font-size: 18px;
            text-align: center;
            margin-bottom: 30px;
        }
        .button {
            display: flex;
            justify-content: center;
        }
        .btn {
            background-color: #1A535C;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .btn:hover {
            background-color: #0A3D44;
        }
        </style>
    """, unsafe_allow_html=True)

    # Tạo giao diện
    st.markdown("<div class='custom-box'>", unsafe_allow_html=True)
    st.markdown("<div class='alert'>HIGH RISK OF DEATH!</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtext'>The mortality rate for this patient is high when they contract COVID-19</div>", unsafe_allow_html=True)
    if st.button('BACK'):
        go_back()
    st.markdown("</div>", unsafe_allow_html=True)
