import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
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

    # Plot feature importance
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))

    sns.barplot(x='Importance', y='Feature', data=top_features, ax=axes[0], palette='plasma')
    for index, value in enumerate(top_features['Importance']):
        axes[0].text(value + 1, index, f'{value:.2f}%', va='center', fontsize=10)
    axes[0].set_title('Top 5 Feature Importances')
    axes[0].set_xlabel('Importance (%)')
    axes[0].set_xlim(0, 100)

    # Plot feature vs. stroke prevalence
    for feature in top_features['Feature']:
        prevalence = balanced_df.groupby(feature)['Stroke'].mean() * 100
        sns.lineplot(x=prevalence.index, y=prevalence.values, ax=axes[1], label=feature)

    axes[1].set_title('Impact of Top Features on Stroke Prevalence')
    axes[1].set_xlabel('Feature Value')
    axes[1].set_ylabel('Stroke Prevalence (%)')
    axes[1].legend(title='Feature')

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file.")
