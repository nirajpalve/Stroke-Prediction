import pandas as pd
import requests
import streamlit as st
from pickle import load
from sklearn.preprocessing import LabelEncoder,StandardScaler
from streamlit_lottie import st_lottie


st.title("STROKE PREDICTION")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_33asonmr.json")



st.sidebar.header('User Input Parameters')

gender = st.sidebar.selectbox('Gender ',['Female','Male'])
age = st.sidebar.number_input('Select Age Range',5,85)
hypertension = st.sidebar.number_input('Do you have Hypertension',0, 1)
heart_disease = st.sidebar.number_input('Do you have Heart Disease',0, 1)
ever_married = st.sidebar.selectbox('Are you Married',['No','Yes'])
work_type = st.sidebar.selectbox('What type of work you do?', ['Private','Self Employed','Goverment Job','Never Worked','Childern'])
Residence_type = st.sidebar.selectbox('Which area do you live?',['Urban','Rural'])
avg_glucose_level = st.sidebar.number_input('Enter your Glucose Level',50.0,300.0)
bmi = st.sidebar.number_input('Enter Your Body Mass Index',0.0,50.0)
smoking_status = st.sidebar.selectbox('Are you Smoker?',['Smokes','Formerly Smoked','Never Smoked','Unknown'])


submit = st.sidebar.button('Submit')


data = {'gender':gender,
        'age':age,
        'hypertension':hypertension,
        'heart_disease':heart_disease,
        'ever_married':ever_married,
        'work_type':work_type,
        'Residence_type':Residence_type,
        'avg_glucose_level':avg_glucose_level,
        'bmi':bmi,
        'smoking_status':smoking_status}

data= pd.DataFrame(data,index=[0])
st.subheader('User Input parameters')
st.write(data)


stroke = pd.read_csv(r"C:\Users\NirajPalve\OneDrive\Desktop\stroke-data.csv")
stroke = stroke.drop(columns=['id','stroke'])
df = pd.concat([data,stroke],axis=0)


encode = ['smoking_status','gender','work_type','Residence_type','ever_married']
df_non_numeric =df.select_dtypes(['object'])
non_numeric_cols = df_non_numeric.columns.values
for col in non_numeric_cols:
    df[col] = LabelEncoder().fit_transform(df[col].values)
    
scelar = StandardScaler()
scelar = scelar.fit_transform(df) 
df = pd.DataFrame(data=scelar,columns=df.columns)   
df = df[:1]



rf_model = load(open(r"C:\Users\NirajPalve\OneDrive\Desktop\rf_model.pkl",'rb'))


def result():
    prediction = rf_model.predict(df)
    if prediction == 0:
        results = "You don't have stroke"
    else:
        results = "You have a stroke"
    return results
        
results = result()
        
if submit is True:
    st.write(results)

st_lottie(lottie_coding, height = 300, key = "coding")
