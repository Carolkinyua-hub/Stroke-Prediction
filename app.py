import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
import joblib

# Streamlit app configuration
st.title('Stroke Prediction and Factor Analysis App')

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    
    # Separate classes and perform downsampling
    majority_class = data[data['Stroke'] == 0]
    minority_class = data[data['Stroke'] == 1]
    majority_downsampled = majority_class.sample(n=len(minority_class), random_state=42)
    balanced_df = pd.concat([minority_class, majority_downsampled])
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

    # Perform Factor Analysis
    fa = FactorAnalysis(n_components=5, random_state=42)
    X_factors = fa.fit_transform(X_scaled)
    
    # Get factor loadings
    factor_loadings = pd.DataFrame(fa.components_.T, columns=[f'Factor {i+1}' for i in range(fa.n_components_)], index=X_balanced.columns)

    # Normalize factor loadings to percentages
    factor_loadings_percentage = factor_loadings * 100
    
    # Display factor loadings
    st.write("Factor Loadings (in Percentages):")
    st.dataframe(factor_loadings_percentage)
    
    # Define the data for pie charts based on factor loadings
    def plot_pie_chart(ax, labels, sizes, title):
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
        ax.axis('equal')
        ax.set_title(title)
    
    # Create pie charts for each factor
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for i in range(5):
        # Extract top features for each factor
        top_features = factor_loadings_percentage[f'Factor {i+1}'].abs().sort_values(ascending=False).head(10)
        labels = top_features.index
        sizes = top_features.values

        # Plot pie chart
        plot_pie_chart(axes[i // 2, i % 2], labels, sizes, f'Factor {i+1}')

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file.")
