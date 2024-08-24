import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Streamlit app configuration
st.title('Stroke Prediction Dashboard')

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)

    # Separate classes
    majority_class = data[data['Stroke'] == 0]
    minority_class = data[data['Stroke'] == 1]

    # Downsample the majority class
    majority_downsampled = majority_class.sample(n=len(minority_class), random_state=42)

    # Combine the minority class with the downsampled majority class
    balanced_df = pd.concat([minority_class, majority_downsampled])

    # Shuffle the dataset to ensure the classes are mixed
    balanced_df = shuffle(balanced_df, random_state=42).reset_index(drop=True)

    # Separate features and target in the balanced dataset
    X_balanced = balanced_df.drop('Stroke', axis=1)  # Features
    y = balanced_df['Stroke']  # Target variable

    # Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    # Load the pre-trained Neural Network model
    model = joblib.load('neural_network_model_selected_features.joblib')

    # Make predictions on the data
    y_pred_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    # Plot confusion matrix
    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Stroke', 'Stroke'])
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    st.pyplot()

    # Plot stroke prevalence distribution
    st.subheader('Stroke Prevalence Distribution')
    plt.figure(figsize=(8, 6))
    sns.histplot(data['Stroke'], kde=False, bins=2, color='skyblue')
    plt.title('Distribution of Stroke')
    plt.xlabel('Stroke')
    plt.ylabel('Count')
    st.pyplot()

    # Plot distribution of stroke-predicted cases for each feature
    st.subheader('Feature Distributions for Predicted Stroke Cases')
    for feature in X_balanced.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=feature, hue='Stroke', multiple='stack', palette='Set2')
        plt.title(f'Distribution of {feature} by Stroke Prediction')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.legend(title='Stroke', labels=['No Stroke', 'Stroke'])
        st.pyplot()

else:
    st.write("Please upload a CSV file.")
