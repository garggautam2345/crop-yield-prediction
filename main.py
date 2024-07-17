import streamlit as st
import pandas as pd
import numpy as np 
import pre as pre
st.markdown(
    """
    <style>
    .title {
        font-size: 50px;
        font-family: 'Arial Black', Gadget, sans-serif;
        color:#7719d4;
    }
    .subtitle {
        font-size: 30px;
        font-family: 'Courier New', Courier, monospace;
        color: #4682b4;
    }
    .custom-text {
        font-size: 20px;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: #2e8b57;
    }
    </style>
    """,
    unsafe_allow_html=True
) 

df=pd.read_csv('yield_df.csv')
df.drop(columns=['Unnamed: 0'],inplace=True)

x_train,x_test,y_train,y_test=pre.pre(df)
x_train_updated,x_test_updated=pre.updated(df,x_train,x_test)
a=pre.model(x_train_updated,y_train)

st.markdown('<p class="title">Crop Yield Production</p>', unsafe_allow_html=True)
# prediction=pre.predict(Area,Item,Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,a)
st.markdown('<p class="subtitle">Input Your Features </p>', unsafe_allow_html=True)

with st.form("input form"):
    options=['Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia',
       'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
       'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 'Brazil',
       'Bulgaria', 'Burkina Faso', 'Burundi', 'Cameroon', 'Canada',
       'Central African Republic', 'Chile', 'Colombia', 'Croatia',
       'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
       'Eritrea', 'Estonia', 'Finland', 'France', 'Germany', 'Ghana',
       'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras',
       'Hungary', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Italy',
       'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 'Lebanon',
       'Lesotho', 'Libya', 'Lithuania', 'Madagascar', 'Malawi',
       'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico',
       'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal',
       'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Norway',
       'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal',
       'Qatar', 'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal',
       'Slovenia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan',
       'Suriname', 'Sweden', 'Switzerland', 'Tajikistan', 'Thailand',
       'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom',
       'Uruguay', 'Zambia', 'Zimbabwe']
    Area=st.selectbox("Area",options)
    valid_options = ['Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Wheat',
       'Cassava', 'Sweet potatoes', 'Plantains and others', 'Yams']
    Item=st.selectbox("enter the item",valid_options)
    Year=st.text_input("Year","2000")
    average_rain_fall_mm_per_year=st.text_input("average_rain_fall_mm_per_year","1485")
    pesticides_tonnes=st.text_input("pesticide_tonnes","121")
    avg_temp=st.text_input("avg_temp","16.57")
    submit_button = st.form_submit_button("Submit")

ans=pre.prediction(Area,Item,Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,a)
st.markdown('<p class="title">based on the data that you had given the production will be </p>', unsafe_allow_html=True)
st.write(ans)
