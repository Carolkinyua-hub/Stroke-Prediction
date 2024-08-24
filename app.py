import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_curve, auc
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

    # Check lengths
    st.write(f"Length of balanced data: {len(balanced_df)}")
    st.write(f"Length of X_balanced: {len(X_balanced)}")
    st.write(f"Length of y: {len(y)}")

    # Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    # Load the pre-trained Neural Network model
    model = joblib.load('neural_network_model_selected_features.joblib')

    # Make predictions on the data
    y_pred_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    # Check lengths again
    st.write(f"Length of X_scaled: {len(X_scaled)}")
    st.write(f"Length of y_pred_prob: {len(y_pred_prob)}")
    st.write(f"Length of y_pred: {len(y_pred)}")

    if len(X_scaled) != len(y_pred):
        st.write("Mismatch between length of predictions and dataset.")
    else:
        # Display stroke prevalence
        st.write(f"Stroke Prevalence: {y.mean() * 100:.2f}%")

        # Plot ROC Curve
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

        # Group data by feature and stroke status and plot heatmaps
        st.subheader('Feature-wise Stroke Status Percentages')

        def plot_feature_heatmap(feature):
            if feature not in X_balanced.columns:
                st.write(f"{feature} is not a valid feature in the dataset.")
                return
            
            # Group and calculate percentages
            feature_data = pd.concat([X_balanced[feature], y], axis=1)
            feature_summary = feature_data.groupby([feature, 'Stroke']).size().unstack(fill_value=0)
            total_counts = feature_summary.sum()
            feature_percentages = feature_summary.div(total_counts, axis=1) * 100

            # Plot heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(feature_percentages, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Percentage (%)'},
                        xticklabels=['No Stroke', 'Stroke'], yticklabels=[0, 1])
            plt.title(f'{feature} Stroke Status Percentages')
            plt.xlabel('Stroke Status')
            plt.ylabel(feature)
            st.pyplot()

        # Plot heatmaps for each feature
        features = X_balanced.columns
        for feature in features:
            plot_feature_heatmap(feature)

else:
    st.write("Please upload a CSV file.")
