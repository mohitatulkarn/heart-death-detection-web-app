import streamlit as st
import pandas as pd
df =pd.read_csv("C:\\Users\\AtulKumar\\Downloads\\pythonproject\\heart_failure_clinical_records_dataset.csv")
st.title('Heart Failure Detection')
def feature_input():
    age=st.number_input("Enter Age range 0-100 in years",0,100)
    anaemia= st.number_input("Enter anaemia either 0 or 1",0,1)
    creatinine_phosphokinase = st.number_input("enter the creatinine_phosphokinase range =30-1300",step=1.,format="%.2f")
    diabetes = st.number_input("Enter diabetes either 0,1",0,1)
    ejection_fraction = st.number_input("ener the ejection_fraction",step=1.,format="%.2f")
    high_blood_pressure = st.number_input("enter the high_blood_pressure ",step=1.,format="%.2f")
    platelets = st.number_input("enter platelets",step=1.,format="%.2f")
    serum_creatinine = st.number_input("Enter the serum_creatinine  range 100-200",step=1.,format="%.2f")
    serum_sodium = st.number_input("Enter the serum_sodium  range= 100-200",step=1.,format="%.2f")
    sex = st.number_input("enter sex 1=male ,0=female")
    smoking = st.number_input("Enter smoking either 0 or 1",0,1)
    time = st.number_input("enter the time ",step=1.,format="%.2f")
    data= {
        'age' :age ,
        'anaemia' : anaemia,
        'creatinine_phosphokinase' : creatinine_phosphokinase,
        'diabetes' : diabetes,
        'ejection_fraction' : ejection_fraction,
        'high_blood_pressure' : high_blood_pressure,
        'platelets' : platelets,
        'serum_creatinine' : serum_creatinine,
        'serum_sodium' : serum_sodium,
        'sex' : sex,
        'smoking': smoking,
        'time' : time 
        }
    feature = pd.DataFrame(data,index=[0])
    return feature

#outlier detection
Q1 =df.quantile(0.25)
Q3= df.quantile(0.75)
IQR = Q3-Q1
df2= df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR))).any(axis=1)]

x=df2.drop("DEATH_EVENT",axis =1)
y=df2['DEATH_EVENT']
input_data = feature_input()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=.95)
from sklearn import linear_model

classifier= linear_model.LogisticRegression()  
classifier.fit(x_train,y_train)
pr=classifier.score(x_test,y_test) 



prediction = classifier.predict(input_data)
if (prediction==0):
    st.write("less chance of death")
elif(prediction ==1):
    st.write("High chance of death")

st.write("Accuracy in  Percentage")
st.write(int(pr*100))
