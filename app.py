import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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

    # Ensure no empty or zero values cause an empty bar
    metrics_df = metrics_df[metrics_df['Percentage'] > 0]

    # Initialize SHAP KernelExplainer
    explainer = shap.KernelExplainer(model.predict, X_scaled)
    shap_values = explainer.shap_values(X_scaled)

    # Plot metrics and SHAP feature importance
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    # Plot metrics
    sns.barplot(x='Percentage', y='Metric', data=metrics_df, ax=axes[0], palette='viridis')
    for index, value in enumerate(metrics_df['Percentage']):
        axes[0].text(value + 1, index, f'{value:.2f}%', va='center', fontsize=10)
    axes[0].set_title('Model Evaluation Metrics in Percentages')
    axes[0].set_xlabel('Percentage (%)')
    axes[0].set_ylabel('Metric')
    axes[0].set_xlim(0, 100)

    # Plot SHAP summary plot (bar)
    shap.summary_plot(shap_values, X_balanced, plot_type="bar", feature_names=X_balanced.columns, show=False)
    axes[1].set_title('SHAP Feature Importance')
    st.pyplot(fig)

    st.write("SHAP Summary Plot")
    shap.summary_plot(shap_values, X_balanced, feature_names=X_balanced.columns)

else:
    st.write("Please upload a CSV file.")
