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
st.set_page_config(page_title="Stroke Prediction Dashboard", layout="wide")
st.title('Stroke Prediction Dashboard')

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)

    # Create a column with descriptive stroke labels
    data['Stroke_Label'] = data['Stroke'].map({0.0: 'No Stroke', 1.0: 'Stroke'})
    
    st.sidebar.header('Data Overview')
    st.sidebar.write("### Dataset Sample")
    st.sidebar.write(data.head())
    st.sidebar.write(f"### Data Shape: {data.shape}")

    # Separate classes
    majority_class = data[data['Stroke'] == 0.0]
    minority_class = data[data['Stroke'] == 1.0]

    # Downsample the majority class
    majority_downsampled = majority_class.sample(n=len(minority_class), random_state=42)

    # Combine the minority class with the downsampled majority class
    balanced_df = pd.concat([minority_class, majority_downsampled])

    # Shuffle the dataset to ensure the classes are mixed
    balanced_df = shuffle(balanced_df, random_state=42).reset_index(drop=True)

    # Separate features and target in the balanced dataset
    X_balanced = balanced_df.drop(['Stroke', 'Stroke_Label'], axis=1)  # Features
    y = balanced_df['Stroke']  # Target variable

    # Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    # Load the pre-trained Neural Network model
    model = joblib.load('neural_network_model_selected_features_.joblib')

    # Make predictions on the data
    y_pred_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    # Convert y_pred to a Pandas Series for mapping
    y_pred_series = pd.Series(y_pred)

    # Distribution of Predicted Stroke Cases
    st.subheader('Distribution of Predicted Stroke Cases')
    prediction_df = pd.DataFrame({
        'True Stroke': data['Stroke_Label'], 
        'Predicted Stroke': y_pred_series.map({0.0: 'No Stroke', 1.0: 'Stroke'})
    })
    prediction_counts = prediction_df['Predicted Stroke'].value_counts().reset_index()
    prediction_counts.columns = ['Predicted Stroke', 'Count']

    fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Predicted Stroke', y='Count', data=prediction_counts, palette='coolwarm', ax=ax_pred)
    for index, value in enumerate(prediction_counts['Count']):
        ax_pred.text(index, value + 1, f'{value}', ha='center', fontsize=10)
    ax_pred.set_title('Predicted Stroke vs Non-Stroke Cases')
    ax_pred.set_xlabel('Predicted Stroke')
    ax_pred.set_ylabel('Count')
    st.pyplot(fig_pred)

    # Compute classification metrics
    report = classification_report(y, y_pred, output_dict=True)

    # Extract metrics dynamically
    metrics = {
        'accuracy': accuracy_score(y, y_pred) * 100,
        'precision_stroke': report.get('1.0', {}).get('precision', 0) * 100,
        'recall_stroke': report.get('1.0', {}).get('recall', 0) * 100,
        'f1_score_stroke': report.get('1.0', {}).get('f1-score', 0) * 100,
        'precision_no_stroke': report.get('0.0', {}).get('precision', 0) * 100,
        'recall_no_stroke': report.get('0.0', {}).get('recall', 0) * 100,
        'f1_score_no_stroke': report.get('0.0', {}).get('f1-score', 0) * 100
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

    # Create dashboard layout for metrics
    st.sidebar.header('Model Metrics')
    st.sidebar.write(f"### Accuracy: {metrics['accuracy']:.2f}%")
    st.sidebar.write(f"### Precision (Stroke): {metrics['precision_stroke']:.2f}%")
    st.sidebar.write(f"### Recall (Stroke): {metrics['recall_stroke']:.2f}%")
    st.sidebar.write(f"### F1 Score (Stroke): {metrics['f1_score_stroke']:.2f}%")
    st.sidebar.write(f"### Precision (No Stroke): {metrics['precision_no_stroke']:.2f}%")
    st.sidebar.write(f"### Recall (No Stroke): {metrics['recall_no_stroke']:.2f}%")
    st.sidebar.write(f"### F1 Score (No Stroke): {metrics['f1_score_no_stroke']:.2f}%")

    # Feature Importance
    st.subheader('Feature Importance')
    fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='plasma')
    for index, value in enumerate(importance_df['Importance']):
        ax_importance.text(value + 1, index, f'{value:.2f}%', va='center', fontsize=10)
    ax_importance.set_title('Permutation Feature Importance in Percentages')
    ax_importance.set_xlabel('Importance (%)')
    ax_importance.set_ylabel('Feature')
    ax_importance.set_xlim(0, 100)
    st.pyplot(fig_importance)

    st.subheader('Impact of Top Features on Stroke Prevalence')
    top_features = importance_df.head(5)
    for feature in top_features['Feature']:
        fig_feature, ax_feature = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=balanced_df[feature], y=balanced_df['Stroke'], ci=None, marker='o', ax=ax_feature)
        ax_feature.set_title(f'Impact of {feature} on Stroke Prevalence')
        ax_feature.set_xlabel(feature)
        ax_feature.set_ylabel('Stroke Prevalence')
        st.pyplot(fig_feature)

else:
    st.write("Please upload a CSV file.")
