from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import pickle
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

st.set_page_config(page_title="FloodWatch", page_icon=":bar_chart:",layout="wide", initial_sidebar_state='expanded')
st.title(" :bar_chart:  FloodWatch : An AutoML tool for flood forecasting and area segmentation and classification using weather data and images")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

def runDL(model):
    loaded_model = load_model(f"./{model}.h5") # default loaded model
    model_scores = {
        'MobileNetV2': 97.4,
        'ResNet50': 79.6,
        'InceptionV3': 95.8,
        'Ensemble': 98.51,
    }

    uploaded_file = st.file_uploader(':file_folder: Choose an image to classify as Flood or Non Flood', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        with st.columns(3)[1]:
            st.image(img, caption='Uploaded Image', width=400)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        pred = np.round(loaded_model.predict(img_array))
        # print(int(pred[0][0]))
        if pred[0][0] == 0:
            st.write("Prediction: **Flood**")
        else:
            st.write("Prediction: **No Flood**")
        with st.expander("**Model Information**"):
            st.write("**Model Accuracy:** ", model_scores[model], "%")

            st.write('')
            st.write('')
            st.write('')
            st.write("**Confusion Matrix:**")
            path = f"./{model}conf_mat.png"
            img = Image.open(path)
            # st.image(img, caption=model, use_column_width=True)
            st.image(img, use_column_width=True)
            st.write('A confusion matrix is a table used in machine learning to assess the performance of a classification model by the summarizing the results of classification. It shows the relation and counts of the actual category of the dataset and predicted category of the model.')

            st.write('')
            st.write('')
            st.write('')
            st.write("**Epochs vs Accuracy/Loss Curve:**")
            st.write('This curve is a clear way of visualizing our progress while training an image classification model. The accuracy/loss of the model against each consecutive iteration or epoch is plotted here while the model was undergoing training. The consecutive points on the line correspond to the values recorded in successive epochs. Generally, a better trained model would have a smooth increase and decrease in its accuracies and losses respectively. The training values indicate how well the model predicted the images that it has been trained with. While, the validation values indicate how well the model predicted the unseen images that it has been provided with after each epoch.')
            hist_path = f"./{model}_history.pkl"
            with open(hist_path, 'rb') as file:
                loaded_history = pickle.load(file)
                
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # Plot Accuracy
            axes[0].plot(loaded_history['accuracy'], label='Training_Accuracy')
            axes[0].plot(loaded_history['val_accuracy'], label='Validation_Accuracy')
            axes[0].set_xlim(0, 20)
            axes[0].legend()
            axes[0].set_xlabel('Epochs')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Epochs vs Accuracy Curve')
            # Plot Loss
            axes[1].plot(loaded_history['loss'], label='Training_Loss')
            axes[1].plot(loaded_history['val_loss'], label='Validation_Loss')
            axes[1].set_xlim(0, 20)
            axes[1].legend()
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Epochs vs Loss Curve')
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.write("Please upload an image file.")

##################################################### MAIN ######################################################################################################
        
# Dropdowns
selected_dl_model = st.selectbox('Select a Deep Learning Model to run your prediction on an uploaded image', ['None', 'MobileNetV2', 'InceptionV3', 'ResNet50', 'Ensemble'])

# Conditionally display model interfaces based on user selection
if selected_dl_model !="None":
    runDL(selected_dl_model)


