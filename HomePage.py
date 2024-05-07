import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings


warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

st.set_page_config(page_title="Crop Prediction", page_icon="https://i.ibb.co/R6kwPmM/download.jpg", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def set_bg_hack_url():    
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://i.ibb.co/XC1c7tR/Picture1.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def main():
    # background image
    set_bg_hack_url()
    # title
    html_temp = """
    <div>
    <h1 style="color:yellow;text-align:center;">  Crop Prediction using ML 🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
 

    col = st.columns(1)[0]

    with col:
        st.subheader("Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("**Nitrogen (1-1000) **", 1, 10000)
        P = st.number_input("**Phosporus (1-1000) **", 1, 10000)
        K = st.number_input("**Potassium (1-1000) **", 1, 10000 )
        temp = st.number_input("**Temperature (0.0 - 100000)**", 0.0, 100000.0)
        humidity = st.number_input("**Humidity in % (0.0 - 100000)**", 0.0, 100000.0)
        ph = st.number_input("**Ph (0.0 - 100000)**", 0.0, 100000.0)
        rainfall = st.number_input("**Rainfall in mm (0.0 - 100000)**", 0.0, 100000.0)


        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        crop_images = {
        "Rice": "https://i.ibb.co/zGpNC8S/rice.jpg",
        "Maize": "https://i.ibb.co/ZHS5HWy/maize.jpg",
        "Jute": "https://i.ibb.co/ZYVVZxC/jute.jpg",
        "Cotton": "https://i.ibb.co/xC0zZM2/cottoon.jpg",
        "Papaya": "https://i.ibb.co/ZNvpLT4/papaya.jpg",
        "Orange ": "https://i.ibb.co/DG2kD1Q/orange.jpg",
        "Apple": "https://i.ibb.co/rcBJX25/apple.jpg",
        "Coconut": "https://i.ibb.co/j4TnFp2/coconut.jpg",
        "Muskmelon": "https://i.ibb.co/Nm3s8Ry/maskmalon.jpg",
        "Watermelon": "https://i.ibb.co/6Z7qz8W/watermelon.jpg",
        "Grapes": "https://i.ibb.co/gtnNSrx/grapes.jpg",
        "Mango": "https://i.ibb.co/C8YyJxx/mango.jpg",
        "Banana": "https://i.ibb.co/WW3N4RV/banana.jpg",
        "Pomegranate": "https://i.ibb.co/0G7CFTT/anar.jpg",
        "Lentil": "https://i.ibb.co/m44ztmm/lentails.jpg",
        "Blackgram": "https://i.ibb.co/1rpbMYY/blackgram.jpg",
        "Mungbean": "https://i.ibb.co/2htwjg0/mungbens.jpg",
        "Mothbeans": "https://i.ibb.co/xgn2JQ4/mothbens.jpg",
        "Pigeonpeas": "https://i.ibb.co/2jLDpFH/pp.jpg",
        "Kidneybeans": "https://i.ibb.co/dKM2kHy/kb.jpg",
        "Chickpea":"https://i.ibb.co/hc6z7Vh/ck.jpg",
              }
        
        if st.button('Predict'):

        
            loaded_model = load_model("D:\Documents\CropPrediction\model.pkl ") 
                    
            prediction = loaded_model.predict(single_pred)
            col.markdown("<div style='background-color: white; padding: 10px;'><h2 style='color: blue; '>Results 🔍</h2></div>", unsafe_allow_html=True)
            predicted_crop = prediction.item().title()
            col.markdown(f"<div style='background-color: white; padding: 10px;'><p style='font-size: 30px; color: brown; font-weight: bold;'>{predicted_crop} are recommended by the A.I for your farm.</p></div>", unsafe_allow_html=True)
            if predicted_crop in crop_images:
                st.image(crop_images[predicted_crop], use_column_width=True, caption=predicted_crop)


            

    #code for html ☘️ 🌾 🌳 👨‍🌾  🍃
    hide_menu_style = """
    <style>
      
       
    .block-container {padding: 2rem 1rem 3rem;}
    
    #MainMenu {visibility: hidden;}
    </style>
    """
    





hide_menu_style = """
       <style>
   .block-container {padding: 2rem 1rem 3rem;}
        #MainMenu {visibility: hidden;}
     
      </style>
       """

st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()

