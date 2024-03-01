import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import segmentation_models as sm
from keras.models import load_model
from keras.utils import custom_object_scope
from keras.preprocessing import image

st.set_page_config(page_title="FloodWatch", page_icon=":bar_chart:",layout="wide", initial_sidebar_state='expanded')
st.title(" :bar_chart:  FloodWatch : An AutoML tool for flood forecasting and area segmentation and classification using weather data and images")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)


def iou_score(y_true, y_pred):
    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))
    iou = intersection / union if union > 0 else 0.0
    return iou


def modify_mask(mask):
    mask = np.expand_dims(mask, axis=2)
    t_mask = np.zeros(mask.shape)
    np.place(t_mask[:, :, 0], mask[:, :, 0] >= 0.5, 1)
    return t_mask

def make_pred_good(pred):
    pred = pred[0][:, :, :]
    pred = modify_mask(pred[:, :, 0])
    pred = np.repeat(pred, 3, 2)
    return pred

def placeMaskOnImg(img, mask):
    color = np.array([161, 205, 255])/255.0
    np.place(img[:, :, :], mask[:, :, :] >= 0.5, color)
    return img

with custom_object_scope({'iou_score': iou_score}):
    model = load_model('./unet.h5')

image_file = st.file_uploader(":file_folder: Upload an image file", type=["png", "jpg", "jpeg"])
if image_file is not None:
        img = image.load_img(image_file, target_size=(224, 224))
        img = np.array(img)[:, :, :3]
        img = img/255.0
        img = np.expand_dims(img, axis = 0)

        fig, axes = plt.subplots(1, 3, figsize=(16, 16))
        axes[0].set_title("Image")
        axes[0].imshow(img[0])
        axes[0].axis('off')
        axes[1].set_title("Predicted Mask")
        pred = make_pred_good(model(img))
        axes[1].imshow(pred)
        axes[1].axis('off')
        axes[2].set_title("Mask on Image")
        axes[2].imshow(placeMaskOnImg(img[0], pred))
        axes[2].axis('off')
        st.pyplot(fig)

        with st.expander("**Model Information**"):
            st.write("Epochs vs Loss/IoU Curve:")
            hist_path = f"./Unet_history.pkl"
            with open(hist_path, 'rb') as file:
                loaded_history = pickle.load(file)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # Plot Loss
            axes[0].plot(loaded_history['loss'], label = 'Training')
            axes[0].plot(loaded_history['val_loss'], '--r', marker = 'o', label = 'Validation')
            axes[0].legend()
            axes[0].set_xlabel('Epochs')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Epochs vs Loss Curve')
            # Plot IoU Score
            axes[1].plot(loaded_history['iou_score'], label = 'Training')
            axes[1].plot(loaded_history['val_iou_score'], '--r', marker = 'o', label = 'Validation')
            axes[1].set_xlim(0, 20)
            axes[1].legend()
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('IoU Score')
            axes[1].set_title('Epochs vs IoU Score Curve')
            plt.tight_layout()
            st.pyplot(fig)

            st.write('An IoU score or Intersection over Union score indicates the ratio of the intersection of the predicted mask area and the actual image area to their combined areas. A higher IoU score indicates that the model was more accurate in segmenting pixels of images into flood and non flood areas.')
else:
    st.write("Please upload an image file.")




