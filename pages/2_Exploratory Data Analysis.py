import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

st.set_page_config(page_title="FloodWatch", page_icon=":bar_chart:",layout="wide", initial_sidebar_state='expanded')
st.title(" :bar_chart:  FloodWatch : An AutoML tool for flood forecasting and area segmentation and classification using weather data and images")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

st.title('Data Acquisition')
st.subheader(f'We have collected the flooding dataset from a reputed source in [Github](https://github.com/n-gauhar/Flood-prediction)')
df = pd.read_csv("./FloodPredictionOriginal.csv") 
flag=0
uploaded_file = st.file_uploader(":file_folder: You can upload a similar new dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    flag=1
if flag==0:
    st.subheader('Default Dataset:')
else:
    st.subheader('Uploaded Dataset:')
st.write(df)



st.markdown('<hr style="border:2px solid #000000">', unsafe_allow_html=True)
st.title('Dataset Information')
desc_df=pd.read_csv('./DataDescription.csv')
info_data = []
for i,column in enumerate(df.columns):
    col_info = {
        'Feature': column,
        'Data Type': str(df[column].dtype),
        'Min Value': df[column].min() if pd.api.types.is_numeric_dtype(df[column]) else None,
        'Max Value': df[column].max() if pd.api.types.is_numeric_dtype(df[column]) else None,
        'Number of Unique Values': df[column].nunique(),
        'Null Count': df[column].isnull().sum(),
        'Description': desc_df['Description'].iloc[i]
    }
    info_data.append(col_info)
st.table(pd.DataFrame(info_data))

st.markdown('<hr style="border:2px solid #000000">', unsafe_allow_html=True)
st.title('Handling Missing Values')
for feature, null_count in df.isnull().sum().items():
    if null_count >0:
        st.write(f'The **{feature}** column is handled by replacing the ***None*** entries with 0')
        df = df.fillna(0)

st.markdown('<hr style="border:2px solid #000000">', unsafe_allow_html=True)
st.title('Feature Importance')
X=df.iloc[:,0:18]
X['Station_Names'] = LabelEncoder().fit_transform(X['Station_Names'])
y = df['Flood?']
y = LabelEncoder().fit_transform(y)
clf = RandomForestClassifier(n_estimators=25, random_state=42)
clf.fit(X, y)
plt.bar(X.columns, clf.feature_importances_)
plt.xticks(rotation=90, ha="right")
plt.xlabel('Features')
plt.ylabel('Importance Score')
st.pyplot(plt)
st.text('The most important feature is the Rainfall Amount.')
st.text('All of the locational features are equally less important.')
st.text(' Thus, Station Number can be considerred for further analysis, while drop the rest.')
st.text('Serial Number column adds no importance and its value is entirely coincidental.')
st.text('Temporal features are also equally less important. So, drop the Period column.')


st.markdown('<hr style="border:2px solid #000000">', unsafe_allow_html=True)
st.title('Data Visualisation')
stations = ['Barisal', 'Bhola', 'Bogra', 'Chandpur','Chittagong (City-Ambagan)', 'Chittagong (IAP-Patenga)', 'Comilla',"Cox's Bazar", 'Dhaka', 'Dinajpur', 'Faridpur', 'Feni', 'Hatiya','Ishurdi', 'Jessore', 'Khepupara', 'Khulna', 'Kutubdia','Madaripur', 'Maijdee Court', 'Mongla', 'Mymensingh', 'Patuakhali','Rajshahi', 'Rangamati', 'Rangpur', 'Sandwip', 'Satkhira','Sitakunda', 'Srimangal', 'Sylhet', 'Tangail', 'Teknaf']
selected_stations = st.multiselect("Select Stations", stations)
year_slider = st.slider('Select a year:', min_value=1948, max_value=2013, step=1, value=1948)
WeatherFeature = st.radio('Select a meteorological feature to filter out:', ['Rainfall', 'Cloud_Coverage', 'Wind_Speed', 'Relative_Humidity', 'Max_Temp', 'Min_Temp', 'Bright_Sunshine'])
annualdf = (
    df[(df.Year == year_slider) & (df.Station_Names.isin(selected_stations))]
    .groupby(['Year', 'Station_Names'])[WeatherFeature].sum().to_frame().sort_values(by='Year')
)
st.write('**Annual Results**:')
col1, col2 = st.columns([4, 6])  # Use st.columns to display both table and bar chart side by side with 40% and 60% width
col1.write(annualdf)
barplotDf = (
    df[(df['Year'] == year_slider) & (df['Station_Names'].isin(selected_stations))]
    .groupby(['Station_Names', 'Year'])[WeatherFeature].mean().to_frame().reset_index().sort_values(by='Year').reset_index(drop=True)
)
bar_fig = px.bar(barplotDf, x='Station_Names', y=WeatherFeature, color='Year', labels={'Station_Names': 'Station Names', WeatherFeature: f'Mean {WeatherFeature}'}, title='Bar Plot:')
col2.plotly_chart(bar_fig)
filtered_data = (
    df[df.Station_Names.isin(selected_stations)]
    .groupby(['Year', 'Station_Names'])[WeatherFeature].sum().to_frame().reset_index().sort_values(by='Year').reset_index(drop=True)
)
line_fig = px.line(filtered_data, x='Year', y=WeatherFeature, color='Station_Names', labels={'Year': 'Year', WeatherFeature: f'Annual {WeatherFeature}'}, title='Station Line Plot (Click on the legends to hide the line)', line_group='Station_Names')
# line_fig.update_layout(legend=dict(traceorder='reversed'))
# line_fig.update_traces(visible='legendonly')
st.plotly_chart(line_fig)


st.markdown('<hr style="border:2px solid #000000">', unsafe_allow_html=True)
st.title('Univariate Analysis')
st.subheader('For Continuous Features-')
selected_col = st.selectbox("Select a Numeric Column:", ['Rainfall', 'Max_Temp', 'Min_Temp', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine'])
skewness = round(df[selected_col].skew(), 2)
st.write(f"Skewness: {skewness}")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df[selected_col], bins=30, color='skyblue', edgecolor='black')
axes[0].set_ylabel('Count')
axes[0].set_title('Histogram')
sns.boxplot(x=df[selected_col], ax=axes[1])
axes[1].set_title('Boxplot')
st.pyplot(fig)
st.write("")
st.write("")
st.write("")
st.subheader('For Categoric Features-')
selected_col = st.selectbox("Select a Categoric Column:", ['Year', 'Month', 'Station_Number', 'Station_Names', 'X_COR', 'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT'])
plt.figure(figsize=(10, 4))
sns.countplot(x=selected_col, data=df[[selected_col]], palette="viridis")
plt.xticks(rotation=90, ha="right")
plt.title(f'Countplot of {selected_col}')
plt.xlabel(selected_col)
plt.ylabel('Count')
st.pyplot(plt)


st.markdown('<hr style="border:2px solid #000000">', unsafe_allow_html=True)
st.title('Bivariate Analysis')
st.subheader('For Categoric Features-')
selected_cat_col1 = st.selectbox("Select the first categorical column:", ['Year', 'Month', 'Station_Number', 'Station_Names', 'X_COR', 'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT'])
selected_cat_col2 = st.selectbox("Select the second categorical column:", ['Year', 'Month', 'Station_Number', 'Station_Names', 'X_COR', 'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT'])
if selected_cat_col1 == selected_cat_col2:
    st.error('Selected columns cannot be same')
else:
    selected_data = df[[selected_cat_col1, selected_cat_col2]]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=selected_data, x=selected_cat_col1, y=selected_cat_col2, palette="viridis")
    plt.xticks(rotation=90, ha="right")
    plt.title(f'Scatterplot of {selected_cat_col1} and {selected_cat_col2}')
    st.pyplot(plt)
st.write("")
st.write("")
st.write("")
st.subheader('For Continuous Features-')
selected_categorical = st.selectbox("Select a categorical column:", ['Year', 'Month', 'Station_Number', 'Station_Names', 'X_COR', 'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT'])
selected_continuous = st.selectbox("Select a continuous column:", ['Rainfall', 'Max_Temp', 'Min_Temp', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine'])
# Group by the selected categorical column and plot the mean of the selected continuous column
fig, ax = plt.subplots(figsize=(10, 6))
df.groupby(selected_categorical)[selected_continuous].mean().plot.bar(ax=ax, fontsize=10)
ax.set_title(f"Bar Chart of {selected_categorical} Vs {selected_continuous}", fontsize=18)
st.pyplot(fig)


st.markdown('<hr style="border:2px solid #000000">', unsafe_allow_html=True)
st.title('Multivariate Analysis or Feature Correlation')
fig=plt.figure(figsize=(20, 20))
df['Station_Names'] = LabelEncoder().fit_transform(df['Station_Names'])
df['Flood?'] = LabelEncoder().fit_transform(df['Flood?'])
sns.heatmap(df.corr(), annot = True, vmin = -1, vmax = 1)
st.pyplot(fig)