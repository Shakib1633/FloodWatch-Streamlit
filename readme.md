<!-- TITLE -->
<h1 align="center"> FloodWatch </h1>
<h3 align="center"> An AutoML Tool for Flood Forecasting and Segmentation using Weather Data and Images </h3> 
</br>


<!-- TABLE OF CONTENTS -->
<h2> Table of Contents </h2>
<ol>
    <li><a href="#overview"> Overview </a></li>
    <li><a href="#prerequisites"> Prerequisites </a></li>
    <li><a href="#datasets"> Datasets </a></li>
    <li><a href="#features"> Features </a></li>
    <li><a href="#installation"> Installation </a></li>
    <li><a href="#run"> Run </a></li>
    <li><a href="#contributors"> Contributors </a></li>
</ol>


<!-- OVERVIEW -->
<h2 id="overview"> Overview </h2>
<p align="justify">
This app has the ability to preprocess rainfall and weather datasets and to perform data augmentation using <b> Generative Adversarial Networks (GANs)</b>. Besides data augmentation, it has the following suite of capabilities:

- **Statistical analyses** of meteorological dataset.
- **Flood forecasting** using 12 different ML algorithms.
- **Image segmentation** using DL methods.
- **Flood image classification** using transfer learning.
- **Geographic Information System (GIS) integration** for spatial analysis.
- **Explainable AI** techniques to enhance the interpretability of model predictions.

A detailed video presentation of the entire web app can be found at 
[this youtube video](https://www.youtube.com/watch?v=iT2rulSI0LM).
</p>


<!-- PREREQUISITES -->
<h2 id="prerequisites"> Prerequisites </h2>
<p align="justify">
The following open source packages are used in this project:

- Streamlit
- Plotly
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn
- Shap
- Numpy
- Tensorflow
- Keras
- Pillow
- Segmentation-models
</p>


<!-- DATASETS -->
<h2 id="datasets"> Datasets </h2>

1. [Flood-prediction](https://github.com/n-gauhar/Flood-prediction) consists of meteorological parameters or features spanning
320 over 12 months from 1949 to 2013 across 33 different stations in Bangladesh.
2. [FloodNet-Supervised_v1.0](https://github.com/BinaLab/FloodNet-Supervised_v1.0) comprises 2343 high-resolution aerial images collected using DJI Mavic Pro drones after Hurricane Harvey, annotated for semantic segmentation tasks.
3. [Flood-area-segmentation](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation) contains 290 images of flood-affected areas along with corresponding mask images indicating water regions.


<!-- FEATURES -->
<h2 id="features"> Features </h2>
<p align="justify">
This app has the following features:

- ML based Flood Prediction from Weather Dataset
- Feature Relevance Analysis for Flood Prediction Using Explainable AI
- Synthetic Weather Data Generation with Tabular GAN
- Classification of Flood Images via DL Models
- Flooded Area Segmentation using DL Models
- Flooded Area Monitoring along with GIS Visualization
</p>


<!-- INSTALLATION -->
<h2 id="installation"> Installation </h2>

```
git clone https://github.com/Shakib1633/FloodWatch-Streamlit.git
cd FloodWatch-Streamlit
pip install streamlit
pip install plotly
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install seaborn
pip install shap
pip install numpy
pip install tensorflow
pip install keras
pip install pillow
pip install segmentation-models
```


<!-- RUN -->
<h2 id="run"> Run </h2>

```
streamlit run Home.py
```


<!-- CONTRIBUTORS -->
<h2 id="contributors"> Contributors </h2>
<p>
All contributors in this project are from <a href="https://mist.ac.bd/department/cse">Department of Computer Science and Engineering</a> <b>@</b> <a href="https://mist.ac.bd/">Military Institute of Science and Technology</a><br>
  
<b>Tasnim Ullah Shakib</b> <br>
Email: <a>t.shakib1633@gmail.com</a> <br>

<b>Tariq Hasan Rizu</b> <br>
Email: <a>tariqhasanr@gmail.com</a> <br>

<b>Ellora Yasi</b> <br>
Email: <a>ellora.yasi@gmail.com</a> <br>

<b>Dr. Nusrat Sharmin</b> <br>
Email: <a>nusrat@cse.mist.ac.bd</a> <br>

<b>Rubyeat Islam</b> <br>
Email: <a>rubyeat88@gmail.com</a> <br>
</p>
