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

    # Pie charts for factors
    def plot_pie_chart(ax, labels, sizes, title):
        sizes = np.clip(sizes, 0, None)  # Ensure sizes are non-negative
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
        ax.axis('equal')
        ax.set_title(title)

    # Data for pie charts
    factor_1 = {
        'labels': ['HeartDiseaseorAttack', 'BMI', 'GenHlth', 'PhysHlth', 'MentHlth', 'Diabetes', 'NoDocbcCost', 'DiffWalk'],
        'sizes': [0.262, 0.243, 0.756, 0.733, 0.493, 0.277, 0.294, 0.587],
        'title': 'Factor 1: General Health Status and Healthcare Access'
    }
    
    factor_2 = {
        'labels': ['HighBP', 'HighChol', 'AnyHealthcare', 'Age', 'Education', 'Income'],
        'sizes': [0, 0, 0, 0, 0.118, 0.037],  # Adjusted sizes for non-empty pie chart
        'title': 'Factor 2: Hypertension, Cholesterol, and Healthcare Access'
    }
    
    factor_3 = {
        'labels': ['Education', 'Income', 'Veggies', 'Smoker'],
        'sizes': [0, 0, 0, 0.070],  # Adjusted sizes for non-empty pie chart
        'title': 'Factor 3: Socioeconomic Status and Lifestyle Factors'
    }
    
    factor_4 = {
        'labels': ['BMI', 'Diabetes', 'HighChol'],
        'sizes': [0.497, 0.370, 0.200],
        'title': 'Factor 4: Metabolic and Obesity-Related Health Issues'
    }
    
    factor_5 = {
        'labels': ['Fruits', 'Sex', 'Smoker', 'DiffWalk'],
        'sizes': [0.216, 0.468, 0, 0.123],  # Adjusted sizes for non-empty pie chart
        'title': 'Factor 5: Lifestyle and Demographic Characteristics'
    }

    # Increase the size of pie charts
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))  # Adjust the figsize for larger charts
    plot_pie_chart(axes[0, 0], factor_1['labels'], factor_1['sizes'], factor_1['title'])
    plot_pie_chart(axes[0, 1], factor_2['labels'], factor_2['sizes'], factor_2['title'])
    plot_pie_chart(axes[1, 0], factor_3['labels'], factor_3['sizes'], factor_3['title'])
    plot_pie_chart(axes[1, 1], factor_4['labels'], factor_4['sizes'], factor_4['title'])
    plot_pie_chart(axes[2, 0], factor_5['labels'], factor_5['sizes'], factor_5['title'])

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file.")
