#Import the necessary libraries
import streamlit as st
import pickle
import pandas as pd
from PIL import Image
from sklearn import *

#Load the model
def load_model():
    with open('best_model.pkl','rb') as file:
        load_data=pickle.load(file)
    return load_data

model_data = load_model()
model = model_data["model"]
mon_encoder = model_data["le_mon"]
ty_encoder = model_data["le_ty"]
polynom = model_data["poly"]

#This is for the title
st.title('Intelligent Waste Predictor System')

#This for describing more about my system
"""Using MACHINE LEARNING, this system will help
determine how well the authority responsible for waste management
need to be prepared in managing the various types of waste that would
probably accumulate uncontrollably in future.At it's Backend this
system uses an estimator(a machine learning algorithm),therefore
the output is not 100% accurate.However the system predicts the
result with a reliable accuracy of about 84%. """

#Here am going to attach a photo
image = Image.open('waste.jpg')
st.image(image,' System crafted with love and passion by Liz Muriithi (@University of Embu)')

#User data
st.header('Provide the Below Information')

#Note
st.markdown('**NB: Based on the data used to train this model,it is assumed that the model was made in the year 2011**')

#Store the input 'MONTH' selected by the user and tell them what they have choosen
month=('May','December', 'March','October', 'June', 'January',
        'November', 'February', 'July', 'April', 'September', 'August')

chosen_month=st.selectbox('Select the month of the year you want your predictions done',month)
st.write('You selected the month of: ',chosen_month)

#Store the input 'YEAR' entered by the user and tell them what they have choosen
year = (2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,
        2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,
        2033,2034,2035,2036,2037,2038,2039,2040,2041,2042,2043,
        2044,2045,2046,2047,2048,2049,2050,2051,2052,2053,2054,
        2055,2056,2057,2058,2059,2060,2061,2062,2063,2064,2065,
        2066,2067,2068,2069,2070,2071,2072,2073,2074,2075,2076,
        2077,2078,2079,2080,2081,2082,2083,2084,2085,2086,2087,
        2088,2089,2090,2091,2092,2093,2094,2095,2096,2097,2098,
        2099,20100)

chosen_year = st.selectbox('Select the year you want your predictions done',year)
st.write('You selected the year: ',chosen_year)

#Store the input 'TYPE' entered by the user and tell them what they have choosen
waste=('Sidewalk Debris','Misc. Garbage', 'Misc. Recycling',
        'Asphalt Debris', 'Yard Waste', 'Recycled Tires', 'Bottle Bill',
        'Scrap Metal', 'Curb Garbage', 'Haz Waste', 'Curb Recycling',
        'E-Waste')

chosen_type=st.selectbox('Select the type of waste you want predictions on',waste)
st.write('You selected the type of waste as: ',chosen_type)

 #Make a dataframe using the user input data
input_data = {'MONTH':chosen_month,'YEAR':chosen_year,'TYPE':chosen_type}
input_df = pd.DataFrame(input_data,index=[0])

#Show the user what they have entered
st.header('Your Input Data')
st.table(input_df)

#Output
st.header('Click "PREDICT" to Get the Output')
ok = st.button("PREDICT")
if ok:
#CLEANING USER INPUT DATA
    Z = input_df[['YEAR','MONTH','TYPE']]
    Z['MONTH'] = mon_encoder.transform(Z['MONTH'])
    Z['TYPE'] = ty_encoder.transform(Z['TYPE'])
    Z[['YEAR']] = preprocessing.normalize(Z[['YEAR']])
    Z = polynom.transform(Z)

     #Predictions
    waste_total = model.predict(Z)

    #This if statement is to classify the results to inform the garbage collectors more about that predicted output
    if(waste_total[0]<0):
        st.markdown(f"[**{waste_total[0]:.2f} TONS**]: (This amount of {chosen_type} could mean that the {chosen_type} waste is/will not be in existence by this time)")
    elif(waste_total[0]<=10):
        st.markdown(f"[**{waste_total[0]:.2f} TONS**]: (This amount of {chosen_type} waste in any environment will be less risky don't bother so much)")
    elif(waste_total[0]>10 and waste_total[0]<=30):
        st.markdown(f"[**{waste_total[0]:.2f} TONS**]: (This amount of {chosen_type} in any environment will be almost risky)")
    else:
        st.markdown(f"[**{waste_total[0]:.2f} TONS**]: (This amount of {chosen_type} in any environment will be very risky! Take caution!!!)")


#Copyright symbol
footer = '<footer style="color: Green;"><p align="center">Created by Liz Muriithi.© 2022 All rights reserved</p></footer>'
st.markdown(footer,unsafe_allow_html=True)
