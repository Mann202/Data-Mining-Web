import streamlit as st
import numpy as np
import pickle
# BỎ THÊM THƯ VIỆN VÀO :V


# LOAD DTS cho MinMaxScaler
with open('./scaler/min_max_scaler.pkl (for KNN).pkl', 'rb') as knn_model_scaler:
    knn_model_age_scaler = pickle.load(knn_model_scaler)
with open('./scaler/minmax_scaler.pkl (for DT, RF, NB).pkl', 'rb') as other_model_scaler:
    other_model_age_scaler = pickle.load(other_model_scaler)


# Set the page title
st.set_page_config(layout="centered",page_title="PROJECT: Predicting high risk of death patients due to COVID-19 using classification algorithms",
                   page_icon=":alembic:")

# Set sidebar
# st.title('PROJECT: CRYPTOCURRENCY PREDICTION USING MACHINE LEARNING')
# Replace with the actual path to your image
image_path = "./images/covid-19.png"
st.sidebar.image(image_path, use_column_width=True)
st.sidebar.title('` Predicting high risk of death COVID-19 patients`')
st.sidebar.markdown('Lecture: **Phd. Cao Thi Nhan** &  \n **Ms.Nguyen Thi Viet Huong**')
st.sidebar.write(' The Team:\n'
                 ' - Bui Viet Dat\n'
                 ' - Bui Quoc Thinh\n'
                 ' - Ly Gia Hieu\n'
                 ' - Do Phuong Nghi\n')
    
# Load models =========================
with open('./models_opt/knn_op.pkl', 'rb') as knn_model_file:
    knn_model = pickle.load(knn_model_file)
with open('./models_opt/gnb_op.pkl', 'rb') as gnb_model_file:
    gnb_model = pickle.load(gnb_model_file)
with open('./models_opt/bnb_op.pkl', 'rb') as bnb_model_file:
    bnb_model = pickle.load(bnb_model_file)
with open('./models_opt/dt_op.pkl', 'rb') as dt_model_file:
    dt_model = pickle.load(dt_model_file)
with open('./models_opt/knn_op.pkl', 'rb') as rf_model_file:
    rf_model = pickle.load(rf_model_file)
# =========================================

# Hàm này dùng để dự đoánp
def predict_high_risk_death(model, features):
    prediction = model.predict(features)
    return prediction


def page_nav():
    prediction = None
    menu = ["Predict COVID-19 patients death", "About the project"]
    st.header(':kiwifruit: PREDICTING HIGH RISK OF DEATH PATIENTS DUE TO COVID-19')
    st.markdown("This project is build based on the historical event **The Covid-19 Pandemic**. Even though the pandemic ended 2 years ago, it still has a big impact on people’s mentality and remains the biggest epidemic event of the century. ")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Clinical disease")
    pneumobia = st.selectbox( 'Do you have Pneumobia?', ('Yes', 'No')) 
    diabetes = st.selectbox( 'Do you have diabetes?', ('Yes', 'No')) 
    hybertension = st.selectbox( 'Do you have hybertension?', ('Yes', 'No')) 
    renalChronal = st.selectbox( 'Do you have renalChronal?', ('Yes', 'No'))
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Medical checkout")

    # Thuộc tính age cần được minMaxScaling
    patient_age = st.number_input('## Age',min_value=1, max_value=150, step=1)
    scaled_age_KNN = knn_model_age_scaler.transform([[patient_age]])
    scaled_age_other = other_model_age_scaler.transform([[patient_age]])
    


    medi_unit = st.selectbox( 'What type of institution of the National Health System that provided the care?', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)) 
    usmer = st.selectbox( 'Do you indicate treatment in medical units of the first or second level??', (1, 2))
    patient_type = st.selectbox( 'Have you ever been hospitalized', ('Yes', 'No'))
    classi_final = st.selectbox( 'Did you have covid test before? if you did select the diagnosed of the test', (1, 2, 3, 4, 5, 6, 7), help = " Values 1-3 mean that the patient was diagnosed with COVID-19 in different degrees. 4 or higher means that the patient is not a carrier of COVID-19 or that the test is inconclusive",)

    # =====LẤY GIÁ TRỊ=====
    # ---------Clinical disease attributes---------
    pneumobia = 1 if pneumobia == 'Yes' else 2
    diabetes = 1 if diabetes == 'Yes' else 2
    hybertension = 1 if hybertension == 'Yes' else 2
    renalChronal = 1 if renalChronal == 'Yes' else 2
    # ---------Medical checkout attributes---------
    patient_type = 1 if patient_type == 'Yes' else 2
    # medi_unit là kiểu int nên khỏi cần chuyển
    # usmer là kiểu int nên khỏi cần chuyển
    # classi_final là kiểu int nên khỏi cần chuyển


    # MEDI_UNIT :v
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
    if 1 <= medi_unit <= 13:
        for i in range(2, 14):
            if i == medi_unit:
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
    if 1 <= classi_final <= 7:
        for i in range(2, 8):
            if i == medi_unit:
                globals()[f"classi_final{i}"] = 1
                break
            else:
                continue

    features_other = np.array([ 
        [usmer, patient_type, pneumobia, scaled_age_other, diabetes, hybertension, renalChronal, medi_unit2, medi_unit3, medi_unit4, medi_unit5, medi_unit6, medi_unit7, medi_unit8, medi_unit9, medi_unit10, medi_unit11, medi_unit12, medi_unit13, classi_final2, classi_final3, classi_final4, classi_final5, classi_final6, classi_final7]], dtype= object)
    features_knn = np.array([ 
        [usmer, patient_type, pneumobia, scaled_age_KNN, diabetes, hybertension, renalChronal, medi_unit2, medi_unit3, medi_unit4, medi_unit5, medi_unit6, medi_unit7, medi_unit8, medi_unit9, medi_unit10, medi_unit11, medi_unit12, medi_unit13, classi_final2, classi_final3, classi_final4, classi_final5, classi_final6, classi_final7]], dtype= object)



    st.markdown("<hr>", unsafe_allow_html=True)
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
            prediction = predict_high_risk_death(knn_model, features_knn)
    if prediction is not None: 
        if prediction[0] == 1:  
            st.markdown("# :red[High risk of death!!]")

        else:
            st.markdown("# :green[Not at high risk of death~]")


page_nav()
