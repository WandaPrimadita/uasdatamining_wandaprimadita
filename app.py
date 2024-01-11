import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model 
svm = pickle.load(open('svc.pkl','rb'))

#load dataset
data = pd.read_csv('bank_customer_churn_dataset.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Bank Customer')


html_layout1 = """
<br>
<div style="background-color:#0000A0 ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Data Customer Bank</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['Naive Bayes','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Bank Customer')


if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Bank Customer</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

#if st.checkbox('EDa'):
#    pr =ProfileReport(data,explorative=True)
#    st.header('**Input Dataframe**')
#    st.write(data)
#    st.write('---')
#    st.header('**Profiling Report**')
#    st_profile_report(pr)

#train test split
X_new = data[['tenure','balance','credit_card','active_member']]
y_new = data['churn']
X = data.drop('churn',axis=1)
y = data['churn']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
X_train, X_test,y_train,y_test = train_test_split(X_new,y_new,test_size=0.20,random_state=42)
svm.fit(X_train, y_train)

pickle.dump(svm, open('SVC_updated.pkl', 'wb'))


#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    tenure = st.sidebar.slider('tenure',0,20,1)
    balance = st.sidebar.slider('balance',0,200,108)
    credit_card = st.sidebar.slider('credit_card',0,140,40)
    active_member = st.sidebar.slider('active_member',0,100,25)
    
    user_report_data = {
        'tenure':tenure,
        'balance':balance,
        'credit_card':credit_card,
        'active_member':active_member,
        
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Customer')
st.write(user_data)

user_result = svm.predict(user_data)
svc_score = accuracy_score(y_test,svm.predict(X_test))

models = [train_test_split]
models_names = ['Naive Bayes']

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu adalah member aktif'
else:
    output ='Tingkatkan lagi transaksi Anda!'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(svc_score*100)+'%')