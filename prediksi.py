
import numpy as np
from util2 import classify, classify2

# Now running the fixed code
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st

# Data Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Neural Network Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
import tensorflow_hub as hub
from tensorflow.keras.callbacks import ModelCheckpoint
import pydot
import graphviz
from tqdm import tqdm
from PIL import ImageOps, Image
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import applications

# Evaluation
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import io

st.set_page_config(page_title="Tuna Loin Quality Prediction", layout="wide")

# Sidebar form
with st.sidebar.form(key='input_form'):
    st.header("Input Data")

    # Define the multiselect options
    options = st.selectbox(
        "What is your selection algorithm",
        ["None", "ResNet", "DenseNet", "InceptionV3"],
    )

    options2 = st.selectbox(
        "What is your goal",
        ["None", "Predict Grade", "Predict Treatment"],
    )

    col1, col2 = st.columns(2)

    # Add a submit button in the first column
    with col1:
        submit = st.form_submit_button('Submit')

    # Add a reset button in the second column
    with col2:
        reset = st.form_submit_button('Reset')

    # Display the selected options if the submit button is clicked
    if submit:
        st.write("Selected algorithm:", options)
        st.write("Selected goal:", options2)

    # Reset the selections if the reset button is clicked
    if reset:
        pass  # st.experimental_rerun() is not supported here

    # upload file
    uploaded_file = st.file_uploader("Select Images file: ", type=['png', 'jpg'])

    b = st.form_submit_button('Show Images')
    if b:
        if uploaded_file is None:
            st.warning("No file uploaded!")
        else:
            st.image(uploaded_file)

col3, col4, col5 = st.columns([1, 10, 1])


with col3:
    st.image('UNPATTI2.PNG', width=130)

with col4:
    st.markdown(
        """
        <div style="color:white;
                    display:fill;
                    border-radius:50px;
                    background-color:black;
                    font-size:200%;
                    font-family:Serif;
                    letter-spacing:2px">
        <p style='padding: 25px; color: White; font-size: 120%; text-align: center;'>
        TUNA LOIN QUALITY PREDICTION
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col5:
    st.image('UNPATTI.PNG', width=120)

try:
    col6, col7 = st.columns([20, 20])
    with col6:
        # Data Augmentation
        train_generator = ImageDataGenerator(rotation_range=90,
                                            width_shift_range=0.05,
                                            height_shift_range=0.05,
                                            shear_range=0.05,
                                            zoom_range=0.05,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            brightness_range=[0.75, 1.25],
                                            rescale=1./255,
                                            validation_split=0.2)

        # 3 kelas grade
        IMAGE_DIR = "fish_dataset/3-Kelas/"
        IMAGE_SIZE = (224, 224)
        BATCH_SIZE = 64
        SEED_NUMBER = 123

        gen_args = dict(target_size=IMAGE_SIZE,
                        color_mode="rgb",
                        batch_size=BATCH_SIZE,
                        class_mode="categorical",
                        classes={"Grade_Alpha": 0, "Grade_Bravo": 1, "Grade_Charley": 2},
                        seed=SEED_NUMBER)

        train_dataset = train_generator.flow_from_directory(
            directory=IMAGE_DIR + "train",
            subset="training", shuffle=True, **gen_args)
        validation_dataset = train_generator.flow_from_directory(
            directory=IMAGE_DIR + "train",
            subset="validation", shuffle=True, **gen_args)

        test_generator = ImageDataGenerator(rescale=1./255)
        test_dataset = test_generator.flow_from_directory(directory=IMAGE_DIR + "test",
                                                        shuffle=False,
                                                        **gen_args)

        # Fix the error in the last part of the code
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
            st.pyplot(plt)
            st.write(image.shape)

    with col7:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = cv2.resize(image, (224, 224))
            st.pyplot(plt)
            st.write(image.shape)

except NameError:
    st.write("No file uploaded")

try:
    col8, col9 = st.columns (2)

    with col8:
        try:
            # Training the Network
            # ResNet
            if options == "ResNet":
                st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">ResNet</div>''', 
                    unsafe_allow_html=True)

                from tensorflow.keras.applications import ResNet152V2
                from tensorflow.keras import layers, Model

                # Define the model
                res_base = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                res_base.trainable = False

                inputs = res_base.input
                x = res_base(inputs, training=False)
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.Dense(128, activation='relu')(x)
                outputs = layers.Dense(3, activation='softmax')(x)
                model_res = Model(inputs, outputs)

                # Capture the model summary
                stream = io.StringIO()
                model_res.summary(print_fn=lambda x: stream.write(x + '\n'))
                summary_str = stream.getvalue()

                # Display the model summary in Streamlit
                st.text(summary_str)


            #DenseNet
            elif options == "DenseNet":
                st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:10px;background-color:#03648a;overflow:hidden;font-weight:900">DenseNet</div>''', 
                    unsafe_allow_html=True)
                    
                from tensorflow.keras.applications import DenseNet201
                from tensorflow.keras import layers, Model

                # Define the model
                input_shape = (224, 224, 3)  # Adjusted input shape

                # Create the DenseNet201 base model
                base_model = DenseNet201(input_shape=input_shape, include_top=False, weights='imagenet')
                base_model.trainable = False

                # Add custom layers on top of the base model
                inputs1 = base_model.input
                x = base_model.output
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.Dense(128, activation='relu')(x)
                outputs1 = layers.Dense(3, activation='softmax')(x)
                model_dense = Model(inputs1, outputs1)

                # Capture the model summary
                stream1 = io.StringIO()
                model_dense.summary(print_fn=lambda x: stream1.write(x + '\n'))
                summary_str1 = stream1.getvalue()

                # Display the model summary
                st.text(summary_str1)


            #InceptionV3
            elif options == "InceptionV3":
                st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">InceptionV3</div>''', 
                    unsafe_allow_html=True)
                    
                from tensorflow.keras.applications import InceptionV3
                from tensorflow.keras import layers, Model

                inception_base = InceptionV3(include_top = False, weights="imagenet", input_shape=(224, 224, 3))
                inception_base.trainable = False


                inputs3 = inception_base.input
                x = inception_base(inputs3, training=False)
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.Dense(128, activation='relu')(x)
                outputs3 = layers.Dense(3, activation = 'softmax')(x)
                model_incep = Model(inputs3, outputs3)

                # Capture the model summary
                stream = io.StringIO()
                model_incep.summary(print_fn=lambda x: stream.write(x + '\n'))
                summary_str = stream.getvalue()

                # Display the model summary
                st.text(summary_str)

        except NameError:
            st.write("No image uploaded")




    if options == "ResNet":
        model_res.compile(optimizer = 'adam',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
        
        checkpoint1 = ModelCheckpoint('model-resnet-q3-1.keras',
                                verbose = 1,
                                save_best_only = True,
                                monitor='val_loss',
                                mode='min')
        
        

    elif options == "DenseNet":
        model_dense.compile(optimizer = 'adam',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
        
        checkpoint2 = ModelCheckpoint('model-densenet-q3-1.keras',
                                verbose = 1,
                                save_best_only = True,
                                monitor='val_loss',
                                mode='min')

        
        
    elif options == "InceptionV3":
        model_incep.compile(optimizer = 'adam',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
        
        checkpoint3 = ModelCheckpoint('model-inception-q3-1.keras',
                                verbose = 1,
                                save_best_only = True,
                                monitor='val_loss',
                                mode='min')





    with col9:
        if options == "ResNet" and options2 == "Predict Grade":
            st.markdown(
                '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Training Evaluation</div>''', 
                unsafe_allow_html=True)
                
            st.image('./Asset Jurnal Q3/Resnet Training Evaluation.png', use_column_width=True)
                
        
        elif options == "DenseNet" and options2 == "Predict Grade":
            st.markdown(
                '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Training Evaluation</div>''', 
                unsafe_allow_html=True)

            st.image('./Asset Jurnal Q3/Densenet Training Evaluation.png', use_column_width=True)


        elif options == "InceptionV3" and options2 == "Predict Grade":
            st.markdown(
                '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Training Evaluation</div>''', 
                unsafe_allow_html=True)

            st.image('./Asset Jurnal Q3/Inception Training Evaluation.png', use_column_width=True)
    

    with col9:
        if options == "ResNet" and options2 == "Predict Treatment":
            st.markdown(
                '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Training Evaluation</div>''', 
                unsafe_allow_html=True)
                
            st.image('./Asset treatment/3 kelas/train eval resnet-treat-3kelas.png', use_column_width=True)
                
        
        elif options == "DenseNet" and options2 == "Predict Treatment":
            st.markdown(
                '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Training Evaluation</div>''', 
                unsafe_allow_html=True)

            st.image('./Asset treatment/3 kelas/train eval densenet-treat-3kelas.png', use_column_width=True)


        elif options == "InceptionV3" and options2 == "Predict Treatment":
            st.markdown(
                '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Training Evaluation</div>''', 
                unsafe_allow_html=True)

            st.image('./Asset treatment/3 kelas/train eval inception-treat-3kelas.png', use_column_width=True)

 
    col10, col11 = st.columns (2)

    with col10:
        
        if options == "ResNet" and options2 == "Predict Grade":
            st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Confusion Matrix</div>''', 
                    unsafe_allow_html=True)
            
            st.image('./Asset Jurnal Q3/heatmap resnet.png', use_column_width=True)
            
    
        elif options == "DenseNet" and options2 == "Predict Grade":
            st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Confusion Matrix</div>''', 
                    unsafe_allow_html=True)

            st.image('./Asset Jurnal Q3/heatmap densenet.png', use_column_width=True)


        elif options == "InceptionV3" and options2 == "Predict Grade":
            st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Confusion Matrix</div>''', 
                    unsafe_allow_html=True)

            st.image('./Asset Jurnal Q3/heatmap inception.png', use_column_width=True)

    
    with col10:
        
        if options == "ResNet" and options2 == "Predict Treatment":
            st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Confusion Matrix</div>''', 
                    unsafe_allow_html=True)
            
            st.image('./Asset treatment/3 kelas/heatmap resnet-3kelas.png', use_column_width=True)
            
    
        elif options == "DenseNet" and options2 == "Predict Treatment":
            st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Confusion Matrix</div>''', 
                    unsafe_allow_html=True)

            st.image('./Asset treatment/3 kelas/heatmap densenet-3kelas.png', use_column_width=True)


        elif options == "InceptionV3" and options2 == "Predict Treatment":
            st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Confusion Matrix</div>''', 
                    unsafe_allow_html=True)

            st.image('./Asset treatment/3 kelas/heatmap inception-3kelas.png', use_column_width=True)


    with col11:
        
        if options == "ResNet" and options2 == "Predict Grade":
            st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Image Prediction</div>''', 
                    unsafe_allow_html=True)
            
            model = load_model('model-resnet-q3-1.h5')

            with open('./model/labels.txt', 'r') as f:
                class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
                f.close()
            

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_column_width=True)
                
                # classify image
                class_name, conf_score = classify(image, model, class_names)
                
                # write classification
                # write classification
                # Create a DataFrame for the results
                data1 = {
                    "Class Name": [class_name],
                    "Confidence Score": [conf_score]
                }

                results_df1 = pd.DataFrame(data1)

                # Display the DataFrame as a table in Streamlit
                st.table(results_df1)

        elif options == "DenseNet" and options2 == "Predict Grade":
            st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Image Prediction</div>''', 
                    unsafe_allow_html=True)
            
            model = load_model('model-densenet-q3-1.h5')
            
            with open('./model/labels.txt', 'r') as f:
                class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
                f.close()
            

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_column_width=True)
                
                # classify image
                class_name, conf_score = classify(image, model, class_names)
                
                # write classification
                # Create a DataFrame for the results
                data2 = {
                    "Class Name": [class_name],
                    "Confidence Score": [conf_score]
                }

                results_df2 = pd.DataFrame(data2)

                # Display the DataFrame as a table in Streamlit
                st.table(results_df2)

        elif options == "InceptionV3" and options2 == "Predict Grade":
            st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Image Prediction</div>''', 
                    unsafe_allow_html=True)
            
            model = load_model('model-inception-q3-1.h5')
        
            with open('./model/labels.txt', 'r') as f:
                    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
                    f.close()
                

            if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, use_column_width=True)
                    
                    # classify image
                    class_name, conf_score = classify(image, model, class_names)
                    
                    # write classification
                    # write classification
                    # Create a DataFrame for the results
                    data3 = {
                        "Class Name": [class_name],
                        "Confidence Score": [conf_score]
                    }

                    results_df3 = pd.DataFrame(data3)

                    # Display the DataFrame as a table in Streamlit
                    st.table(results_df3)
    
    with col11:
        
        if options == "ResNet" and options2 == "Predict Treatment":
            st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Image Prediction</div>''', 
                    unsafe_allow_html=True)
            
            model2 = load_model('model-resnet-treat-3kelas.h5')
            
            with open('./model/labels2.txt', 'r') as f:
                class_names2 = [a[:-1].split(' ')[1] for a in f.readlines()]
                f.close()


            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_column_width=True)
                
                # classify image
                class_name2, conf_score2 = classify2(image, model2, class_names2)
                
                # write classification
                # write classification
                # Create a DataFrame for the results
                data1 = {
                    "Class Name": [class_name2],
                    "Confidence Score": [conf_score2]
                }

                results_df1 = pd.DataFrame(data1)

                # Display the DataFrame as a table in Streamlit
                st.table(results_df1)

        elif options == "DenseNet" and options2 == "Predict Treatment":
            st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Image Prediction</div>''', 
                    unsafe_allow_html=True)
            
            model2 = load_model('model-densenet-treat-3kelas.h5')
            
            with open('./model/labels2.txt', 'r') as f:
                class_names2 = [a[:-1].split(' ')[1] for a in f.readlines()]
                f.close()


            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_column_width=True)
                
                # classify image
                class_name2, conf_score2 = classify2(image, model2, class_names2)
                
                # write classification
                # Create a DataFrame for the results
                data2 = {
                    "Class Name": [class_name2],
                    "Confidence Score": [conf_score2]
                }

                results_df2 = pd.DataFrame(data2)

                # Display the DataFrame as a table in Streamlit
                st.table(results_df2)

        elif options == "InceptionV3" and options2 == "Predict Treatment":
            st.markdown(
                    '''<div style="padding:20px;color:white;margin:0;font-size:150%;text-align:center;display:fill;border-radius:5px;background-color:#03648a;overflow:hidden;font-weight:800">Image Prediction</div>''', 
                    unsafe_allow_html=True)
            
            model2 = load_model('model-inception-treat-3kelas.h5')
        
                
            with open('./model/labels2.txt', 'r') as f:
                    class_names2 = [a[:-1].split(' ')[1] for a in f.readlines()]
                    f.close()


            if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, use_column_width=True)
                    
                    # classify image
                    class_name2, conf_score2 = classify2(image, model2, class_names2)
                    
                    # write classification
                    # write classification
                    # Create a DataFrame for the results
                    data3 = {
                        "Class Name": [class_name2],
                        "Confidence Score": [conf_score2]
                    }

                    results_df3 = pd.DataFrame(data3)

                    # Display the DataFrame as a table in Streamlit
                    st.table(results_df3)

except ValueError:
    st.error("")

# License: This code is licensed under the MIT License.

col12, col13, col14, col15, col16 = st.columns (5)

with col13:
    st.image('lisensi.PNG',width=900)
