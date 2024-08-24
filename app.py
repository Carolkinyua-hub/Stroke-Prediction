import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Streamlit app configuration
st.title('Stroke Prediction App')

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

    # Display stroke prevalence
    st.write(f"Stroke Prevalence: {y.mean() * 100:.2f}%")

    # Plot ROC Curve
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    st.subheader('ROC Curve')
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    st.pyplot()

    # Confusion Matrix
    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['No Stroke', 'Stroke'], 
                yticklabels=['No Stroke', 'Stroke'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    st.pyplot()

    # Distribution plots for each feature
    st.subheader('Feature Distribution for Predicted Stroke Cases')

    # Function to plot distribution of a feature based on predictions
    def plot_feature_distribution(feature):
        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature][y_pred == 1], kde=True, label='Stroke Predicted', color='green', bins=20)
        sns.histplot(data[feature][y_pred == 0], kde=True, label='No Stroke Predicted', color='red', bins=20)
        plt.title(f'Distribution of {feature} for Predicted Stroke Cases')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.legend()
        st.pyplot()

    # Plot distributions for each binary feature
    binary_features = X_balanced.columns
    for feature in binary_features:
        plot_feature_distribution(feature)

else:
    st.write("Please upload a CSV file.")
