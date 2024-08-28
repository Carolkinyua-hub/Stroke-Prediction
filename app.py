import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

# Streamlit app configuration
st.title('Stroke Prediction Model Interpretability')

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)

    # Separate classes and downsample
    majority_class = data[data['Stroke'] == 0]
    minority_class = data[data['Stroke'] == 1]
    majority_downsampled = majority_class.sample(n=len(minority_class), random_state=42)
    balanced_df = pd.concat([minority_class, majority_downsampled])
    balanced_df = shuffle(balanced_df, random_state=42).reset_index(drop=True)

    # Separate features and target
    X_balanced = balanced_df.drop('Stroke', axis=1)
    y = balanced_df['Stroke']

    # Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    # Load the pre-trained Neural Network model
    model = joblib.load('neural_network_model_selected_features.joblib')

    # Make predictions
    y_pred = model.predict(X_scaled)

    # Classification report
    report = classification_report(y, y_pred, output_dict=True)
    st.write("### Classification Report:")

    # Create a DataFrame for the classification report
    report_df = pd.DataFrame(report).transpose().reset_index()
    report_df = report_df.rename(columns={'index': 'Metric'})

    # Filter relevant metrics
    metrics = ['precision', 'recall', 'f1-score']
    filtered_df = report_df[report_df['Metric'].isin(metrics)]

    # Ensure 'support' column is not dropped if it exists
    if 'support' in filtered_df.columns:
        filtered_df = filtered_df.drop('support', axis=1)

    # Bar plot for classification report
    fig, ax = plt.subplots(figsize=(10, 6))
    filtered_df.set_index('Metric').plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_title('Classification Report Metrics')
    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    ax.set_xticklabels(['No Stroke (0.0)', 'Stroke (1.0)'], rotation=0)
    plt.legend(title='Metric')
    st.pyplot(fig)

    # Permutation importance
    results = permutation_importance(model, X_scaled, y, scoring='accuracy', n_repeats=10, random_state=42)
    importances = results.importances_mean * 100
    features = X_balanced.columns

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Filter top 5 important features
    top_features = importance_df.head(5)

    # Compute correlation matrix for features with stroke prevalence
    correlation_matrix = balanced_df[top_features['Feature'].tolist() + ['Stroke']].corr()

    # Create plots
    fig, axes = plt.subplots(len(top_features) + 2, 1, figsize=(12, 4 * (len(top_features) + 2)))

    # Plot top 5 feature importances
    sns.barplot(x='Importance', y='Feature', data=top_features, ax=axes[0], palette='plasma')
    for index, value in enumerate(top_features['Importance']):
        axes[0].text(value + 1, index, f'{value:.2f}%', va='center', fontsize=10)
    axes[0].set_title('Top 5 Feature Importances')
    axes[0].set_xlabel('Importance (%)')
    axes[0].set_xlim(0, 100)

    # Plot correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[-1], fmt=".2f")
    axes[-1].set_title('Correlation Heatmap of Top Features and Stroke')

    plt.tight_layout()
    st.pyplot(fig)

    # Additional insights
    st.write("### Additional Insights:")

    # Distribution of top features
    st.write("#### Distribution of Top Features")
    for feature in top_features['Feature']:
        fig, ax = plt.subplots()
        sns.histplot(balanced_df[feature], kde=True, ax=ax)
        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    # Health impact of top features (if relevant data available)
    if 'HealthMetric' in balanced_df.columns:
        st.write("#### Health Impact of Top Features")
        for feature in top_features['Feature']:
            fig, ax = plt.subplots()
            sns.boxplot(x='Stroke', y=feature, data=balanced_df, ax=ax)
            ax.set_title(f'{feature} by Stroke Status')
            ax.set_xlabel('Stroke')
            ax.set_ylabel(feature)
            st.pyplot(fig)

else:
    st.write("Please upload a CSV file.")
