import streamlit as st
# import sys
# import os
# sys.path.insert(1, "C:/ProgramData/anaconda3/envs/MachineLearning/Lib/site-packages/streamlit_option_menu")
from streamlit_option_menu import option_menu
from PIL import Image
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array

# path = os.pa/th.dirname(__file__)
# print(path)
# image_directory = path+'/local_hospital.png'

# image_directory = "local_hospital.png"
# image = Image.open(image_directory)

# ../FSnm.png
# C:\ProgramData\anaconda3\envs\MachineLearning\Lib\site-packages\streamlit_option_menu
# C:/ProgramData/anaconda3/envs/MachineLearning/Lib/site-packages


# st.set_page_config(page_title='E Health Care', page_icon = image, layout = 'wide', initial_sidebar_state = 'auto')
# favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)


with st.sidebar:
    
    selected = option_menu('E Health Care', ['Disease Prediction'],
        icons=['activity'],
        default_index=0,
        styles={
        "container": {"background-color": "#B2DDED"}
        # "container": {"background-color": "#B2DDED"}
        # "container": {"background-color": "#B2DDED"}
        }
        ) 


if selected == 'Disease Prediction': 
    # Create disease class and load ML model
    disease_model = DiseaseModel()
    disease_model.load_xgboost('D:/dp/xgboost_model.json')

    # Title
    st.write('# Disease Prediction using Machine Learning')
    
   

    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)

    X = prepare_symptoms_array(symptoms)

    # Trigger XGBoost model
    if st.button('Predict'): 
        # Run the model with the python script
        
        prediction, prob = disease_model.predict(X)
        st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')


        tab1, tab2= st.tabs(["Description", "Precautions"])

        with tab1:
            st.write(disease_model.describe_predicted_disease())

        with tab2:
           
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
               
                st.write(f'{i+1}. {precautions[i]}')