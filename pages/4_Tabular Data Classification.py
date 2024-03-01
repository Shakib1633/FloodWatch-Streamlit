import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
# from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="FloodWatch", page_icon=":bar_chart:",layout="wide", initial_sidebar_state='expanded')
st.title(" :bar_chart:  FloodWatch : An AutoML tool for flood forecasting and area segmentation and classification using weather data and images")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)


def runML(selected_ml_model):
    st.title(selected_ml_model)

    data = pd.read_csv("./Required_Feature_Dataset.csv") #maybe replaced if user chooses to upload a new dataset
    flag=0
    uploaded_file = st.file_uploader(":file_folder: Upload a similar new dataset (CSV format)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        flag=1

    if flag==0:
        st.write('Default Dataset:')
    else:
        st.write('Uploaded Dataset:')
    st.write(data)

    st.write('')
    st.write('')
    st.write('Provide your input feature values:')
    year = st.slider("Year", 1948, 2030, 1948)
    month = st.slider("Month", 1, 12, 1)
    max_tmp = st.slider("Max_Temp", 21.6, 44.0, 21.6)
    min_tmp = st.slider("Min_Temp", 6.2, 28.1, 6.2)
    rainfall = st.slider("Rainfall", 0, 2072, 0)
    rel_hum = st.slider("Relative_Humidity", 34, 97, 34)
    wind_spd = st.slider("Wind_Speed", 0.0, 11.2, 0.0)
    cloud_cvrg = st.slider("Cloud_Coverage", 0.0, 7.9, 0.0)
    sunshine = st.slider("Bright_Sunshine", 0.0, 11.0, 0.0)

    x = data.iloc[:,0:9]
    y = data['Flood?']

    minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
    minmax.fit(x).transform(x)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    model = selected_ml_model

    if st.button('Predict'):  
        if (model=="Ada Boost"):
            model = AdaBoostClassifier()
        elif (model=="Decision Tree"):
            model = DecisionTreeClassifier()
        elif (model=="KNN"):
            model = KNeighborsClassifier()
        elif (model=="Ensemble"):
            ada_clf = AdaBoostClassifier()
            dec_clf = DecisionTreeClassifier()
            knn_clf = KNeighborsClassifier()
            model =  VotingClassifier(estimators=[('ada', ada_clf), ('dec', dec_clf), ('knn', knn_clf)], voting='hard')
            
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        f1= f1_score(y_test,predictions)
        st.write(f"Model Accuracy: **{round(accuracy,3)*100}** %")
        st.write(f"Model F1 Score: **{round(f1,3)*100}** %")
        
        data = list(zip([year], [month], [max_tmp], [min_tmp], [rainfall], [rel_hum], [wind_spd], [cloud_cvrg], [sunshine]))
        array = np.array(data)
        minmax.transform(array)
        prediction = model.predict(array)
        # prediction = model.predict_proba(array)
        # if (prediction[0][1]==1):
        if (prediction[0]>0.5):
            st.write("Flood Prediction based on your selected feature values: **Yes**")
        else:
            st.write("Flood Prediction based on your selected feature values: **No**")

        st.write('')
        st.write('')
        cm = confusion_matrix(y_test, predictions)
        st.write("Confusion Matrix:")
        with st.expander("See More"):
            st.write("A confusion matrix is a table used in machine learning to assess the performance of a classification model by the summarizing the results of classification. It shows the relation and counts of the actual category of the dataset and predicted category of the model.")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 16}, ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        st.pyplot(fig)

##################################################### MAIN ######################################################################################################
        
# Dropdowns
selected_ml_model = st.selectbox('Select an ML Model to run your prediction', ['None', 'Ada Boost', 'Decision Tree', 'KNN', 'Ensemble'])

# Conditionally display ML model interface based on user selection
if selected_ml_model !="None":
    runML(selected_ml_model)

