import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import random
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

st.set_page_config(page_title="FloodWatch", page_icon=":bar_chart:",layout="wide", initial_sidebar_state='expanded')
st.title(" :bar_chart:  FloodWatch : An AutoML tool for flood forecasting and area segmentation and classification using weather data and images")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)


def download_button(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv"> :file_folder: **Download CSV file**</a>'
    st.markdown(href, unsafe_allow_html=True)
    return

def CTGAN(data,year):
  generator=[None] * 13
  for i in range(1,13):
    name = f'./generator_{i}.h5'
    generator[i] = load_model(name)

  scaler = StandardScaler()
  scaler.fit(data[['Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine']])
  data[['Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine']]=scaler.transform(data[['Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine']])

  # Create an empty DataFrame to store the generated data
  generated_data_df = pd.DataFrame(columns=['Year', 'Month', 'Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine', 'Station_Number', 'Flood?'])

  #iterate over all 12 months and generate the data for that month
  i=1
  for month in data['Month'].unique():
      # Filter the data for the current station number
      station_data = data[data['Month'] == month]

      # Drop the specified columns
      drop_columns = ['Year', 'Month', 'Station_Number']
      station_data = station_data.drop(columns=drop_columns)

      # Generate new synthetic data for all stations 
      num_samples = (int(year) - 2013) * 33 #for each month, n years of entry for all 33 stations= n*33 entry per month
      noise = np.random.normal(0, 1, size=(num_samples, 100))
      generated_data = generator[i].predict(noise)
      i=i+1

      # Post-process generated data if necessary
      #scale the flood target variable
      scaled_flood = ((generated_data[:, -1]-generated_data[:, -1].min())/(generated_data[:, -1].max()-generated_data[:, -1].min()) * (1 - 0)) + 0
      generated_data[:, -1] = scaled_flood
      generated_data[:, -1] = np.where(generated_data[:, -1] < 0.3, 0, 1)

      # Scale the other parameter values to match the range of the original dataset
      for i in range(0,7):
        generated_data[:, i] = ((generated_data[:, i] - generated_data[:, i].min()) / (generated_data[:, i].max() - generated_data[:, i].min()) * (station_data[station_data.columns[i]].max() - station_data[station_data.columns[i]].min())) + station_data[station_data.columns[i]].min()

      # Create a DataFrame for the generated data
      generated_df = pd.DataFrame(generated_data, columns=['Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine', 'Flood?'])

      generated_df['Month'] = [month] * len(generated_df)
      # Initialize lists to store data
      year_list = []
      station_list = []
      # Iterate over years, station numbers
      for station_number in data['Station_Number'].unique():
        year_list.extend(list(range(2014, int(year)+1 )))
        station_list.extend([station_number] * (int(year)-2013))
      # Create new columns in the DataFrame
      generated_df['Year'] = year_list
      generated_df['Station_Number'] = station_list
      # Rearrange columns if needed
      generated_df = generated_df[['Year', 'Month', 'Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine', 'Station_Number', 'Flood?']]
      generated_data_df = pd.concat([generated_data_df, generated_df], ignore_index=True)

  generated_data_df[['Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine']]=scaler.inverse_transform(generated_data_df[['Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine']])
  generated_data_df = generated_data_df.sort_values(by=['Station_Number','Year','Month'])
  st.dataframe(generated_data_df)

  # Add a download button to download the DataFrame as a CSV file
  download_button(generated_data_df, 'CTGAN_dataset')

  df = pd.DataFrame({
      "Rainfall": generated_data_df["Rainfall"],
      "Month": generated_data_df["Month"],
      "Flood?": generated_data_df["Flood?"],
  })

  st.write('')
  st.write('')
  st.subheader('**Distribution of the Synthetic Data:**')
  fig = sns.catplot(data=df, y="Rainfall", x="Month", hue="Flood?")
  plt.title("Rainfall Vs Month of the Year")
  st.pyplot(fig)
  return 

def Randomise(data,year):
  generated_data_df = pd.DataFrame(columns=['Year', 'Month', 'Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine', 'Station_Number', 'Flood?'])
  for month in data['Month'].unique():
      station_data = data[data['Month'] == month]
      drop_columns = ['Year', 'Month', 'Station_Number']
      station_data = station_data.drop(columns=drop_columns)
      num_samples = (int(year) - 2013) * 33

      min_max_values = {
        "Max_Temp": (data['Max_Temp'].max(), data['Max_Temp'].min()),
        "Min_Temp": (data['Min_Temp'].max(), data['Min_Temp'].min()),
        "Rainfall": (data['Rainfall'].max(), data['Rainfall'].min()),
        "Relative_Humidity": (data['Relative_Humidity'].max(), data['Relative_Humidity'].min()),
        "Wind_Speed": (data['Wind_Speed'].max(), data['Wind_Speed'].min()),
        "Cloud_Coverage": (data['Cloud_Coverage'].max(), data['Cloud_Coverage'].min()),
        "Bright_Sunshine": (data['Bright_Sunshine'].max(), data['Bright_Sunshine'].min())
      }
      generated_data = []
      for i in range(num_samples):
          row = {}
          for feature_name, (min_value, max_value) in min_max_values.items():  # Unpack the values correctly
              row[feature_name] = random.uniform(min_value, max_value)
          row["Flood?"] = random.randint(0, 1)
          generated_data.append(row)

      generated_df = pd.DataFrame(generated_data, columns=['Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine', 'Flood?'])
      generated_df['Month'] = [month] * len(generated_df)
      year_list = []
      station_list = []
      for station_number in data['Station_Number'].unique():
        year_list.extend(list(range(2014, int(year)+1 )))
        station_list.extend([station_number] * (int(year)-2013))
      generated_df['Year'] = year_list
      generated_df['Station_Number'] = station_list
      generated_df = generated_df[['Year', 'Month', 'Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine', 'Station_Number', 'Flood?']]
      generated_data_df = pd.concat([generated_data_df, generated_df], ignore_index=True)
      
  generated_data_df = generated_data_df.sort_values(by=['Station_Number','Year','Month'])
  st.dataframe(generated_data_df)
  download_button(generated_data_df, 'Random_dataset')

  df = pd.DataFrame({
      "Rainfall": generated_data_df["Rainfall"],
      "Month": generated_data_df["Month"],
      "Flood?": generated_data_df["Flood?"],
  })

  st.write('')
  st.write('')
  st.subheader('**Distribution of the Synthetic Data:**')
  fig = sns.catplot(data=df, y="Rainfall", x="Month", hue="Flood?")
  plt.title("Rainfall Vs Month of the Year")
  st.pyplot(fig)
  return 
   

def runGenerator(selected_gan_model):
  st.title(selected_gan_model)

  data = pd.read_csv("./Required_Feature_Dataset.csv") #maybe replaced if user chooses to upload a new dataset
  flag=0
  uploaded_file = st.file_uploader(":file_folder: Upload a similar new dataset (CSV format)", type=["csv"])
  if uploaded_file is not None:
      data = pd.read_csv(uploaded_file)
      flag=1

  if flag==0:
      st.write('**Default Dataset**:')
  else:
      st.write('**Uploaded Dataset**:')
  st.write(data)

  st.write('How many years of data do you want to generate?')
  year = st.slider("Year", 2014, 2030, 2014)
  if st.button('Generate New Data'):
      st.write('')
      st.write('')
      st.write('')
      st.write('**Synthetic Dataset:**')
      if (selected_gan_model=="CTGAN"):
          CTGAN(data,year)
      elif (selected_gan_model=="Random"):
          Randomise(data,year)

  return
          

##################################################### MAIN ######################################################################################################
selected_gan_model = st.selectbox('Select an augmentation technique to generate synthetic data', ['None', 'CTGAN', 'Random'])
if selected_gan_model != 'None':
   runGenerator(selected_gan_model)
  