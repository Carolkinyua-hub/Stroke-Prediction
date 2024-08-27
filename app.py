import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import permutation_importance

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

    # Compute classification metrics
    report = classification_report(y, y_pred, output_dict=True)
    st.write("Classification Report Keys:", report.keys())

    # Determine positive label (assuming binary classification)
    positive_labels = [label for label in report if label not in ['accuracy', 'macro avg', 'weighted avg']]
    
    if positive_labels:
        positive_label = positive_labels[0]
        metrics = {
            'accuracy': accuracy_score(y, y_pred) * 100,
            'precision': report[positive_label]['precision'] * 100,
            'recall': report[positive_label]['recall'] * 100,
            'f1-score': report[positive_label]['f1-score'] * 100
        }
    else:
        st.write("Positive label not found in the report.")
        metrics = {
            'accuracy': accuracy_score(y, y_pred) * 100,
            'precision': 0,
            'recall': 0,
            'f1-score': 0
        }

    # Create DataFrame for metrics
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Percentage'])
    metrics_df = metrics_df.sort_values(by='Percentage', ascending=False)

    # Compute permutation importance
    results = permutation_importance(model, X_scaled, y, scoring='accuracy', n_repeats=10, random_state=42)
    importances = results.importances_mean

    # Convert permutation importances to percentages
    importances_percentage = importances * 100
    features = X_balanced.columns  # Feature names

    # Create DataFrame for permutation importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances_percentage
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot metrics and feature importance
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    # Plot metrics
    sns.barplot(x='Percentage', y='Metric', data=metrics_df, ax=axes[0], palette='viridis')
    for index, value in enumerate(metrics_df['Percentage']):
        axes[0].text(value + 1, index, f'{value:.2f}%', va='center', fontsize=10)
    axes[0].set_title('Model Evaluation Metrics in Percentages')
    axes[0].set_xlabel('Percentage (%)')
    axes[0].set_ylabel('Metric')
    axes[0].set_xlim(0, 100)

    # Plot feature importance
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=axes[1], palette='plasma')
    for index, value in enumerate(importance_df['Importance']):
        axes[1].text(value + 1, index, f'{value:.2f}%', va='center', fontsize=10)
    axes[1].set_title('Permutation Feature Importance in Percentages')
    axes[1].set_xlabel('Importance (%)')
    axes[1].set_ylabel('Feature')
    axes[1].set_xlim(0, 100)

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file.")
