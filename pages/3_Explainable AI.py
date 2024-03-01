import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="FloodWatch", page_icon=":bar_chart:",layout="wide", initial_sidebar_state='expanded')
st.title(" :bar_chart:  FloodWatch : An AutoML tool for flood forecasting and area segmentation and classification using weather data and images")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

data = pd.read_csv("./Required_Feature_Dataset.csv") 
flag=0
uploaded_file = st.file_uploader(":file_folder: You can upload a similar new dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    flag=1

if flag==0:
    st.subheader('Default Dataset:')
else:
    st.subheader('Uploaded Dataset:')
st.write(data)

X = data.iloc[:,0:10]
y = data['Flood?']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Initialize explainer object with the model and the data
explainer = shap.Explainer(model, X_train)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test,check_additivity=False)

# Plot SHAP summary plots
st.markdown('<hr style="border:2px solid #000000">', unsafe_allow_html=True)
fig, ax = plt.subplots() 
shap.summary_plot(shap_values, X, class_names=['Flood', 'No Flood'], show=False)  
ax.set_title("SHAP Summary Plot (Bar)")
# ax.set_xlabel("SHAP Value")
ax.set_ylabel("Feature Name")
st.pyplot(fig)
st.write("The size of barchart shows the relative importance of each feature on the flood or non-flood result")

st.markdown('<hr style="border:2px solid #000000">', unsafe_allow_html=True)
fig, ax = plt.subplots()
shap.summary_plot(shap_values[1], X_test, class_names=['Flood', 'No Flood'])
ax.set_title("SHAP Summary Plot (Point)")
ax.set_xlabel("SHAP Value (impact of feature value on Flood result)")
ax.set_ylabel("Feature Name")
st.pyplot(fig)
st.write("It is evident from the plot that lower rainfall values negitively correlates with the presence of flood, while higher rainfall values positively correlates with the presence of flood.")

st.markdown('<hr style="border:2px solid #000000">', unsafe_allow_html=True)
fig, ax = plt.subplots()
shap.summary_plot(shap_values[0], X_test, class_names=['Flood', 'No Flood'])
ax.set_title("SHAP Summary Plot (Point)")
ax.set_xlabel("SHAP Value (impact of feature value on No Flood result)")
ax.set_ylabel("Feature Name")
st.pyplot(fig)
st.write('''It is evident from the plot that lower rainfall values postively correlates with the absence of flood, while higher rainfall values negatively correlates with the absence of flood. 
         \n The plot for No Flood output is a reverse of the plot for Flood output.''')