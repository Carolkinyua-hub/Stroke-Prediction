import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

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

    # Ensure we're working with the correct label for the positive class (assuming '1' is the positive class)
    if '1' in report:
        positive_label = '1'
    else:
        positive_label = list(report.keys())[1]  # Fallback to the first label (excluding 'accuracy')

    metrics = {
        'accuracy': accuracy_score(y, y_pred) * 100,
        'precision': report[positive_label]['precision'] * 100,
        'recall': report[positive_label]['recall'] * 100,
        'f1-score': report[positive_label]['f1-score'] * 100
    }

    # Create DataFrame for metrics
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Percentage'])
    metrics_df = metrics_df.sort_values(by='Percentage', ascending=False)

    # Permutation importance
    results = permutation_importance(model, X_scaled, y, scoring='accuracy', n_repeats=10, random_state=42)
    importances = results.importances_mean * 100
    features = X_balanced.columns

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Plot metrics and feature importance
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # Plot metrics
    sns.barplot(x='Percentage', y='Metric', data=metrics_df, ax=axes[0], palette='viridis')
    for index, value in enumerate(metrics_df['Percentage']):
        axes[0].text(value + 1, index, f'{value:.2f}%', va='center', fontsize=10)
    axes[0].set_title('Model Evaluation Metrics in Percentages')
    axes[0].set_xlabel('Percentage (%)')
    axes[0].set_xlim(0, 100)

    # Plot feature importance
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=axes[1], palette='plasma')
    for index, value in enumerate(importance_df['Importance']):
        axes[1].text(value + 1, index, f'{value:.2f}%', va='center', fontsize=10)
    axes[1].set_title('Permutation Feature Importance in Percentages')
    axes[1].set_xlabel('Importance (%)')
    axes[1].set_xlim(0, 100)

    # Plot partial dependence
    display = PartialDependenceDisplay.from_estimator(model, X_scaled, [0, 1], ax=axes[2])
    axes[2].set_title('Partial Dependence Plot')
    axes[2].set_xlabel('Feature Value')
    axes[2].set_ylabel('Predicted Probability')

    plt.tight_layout()
    st.pyplot(fig)

    # Textual Insights and Recommendations
    st.write("**Insight:** Age and Blood Pressure are critical features influencing stroke likelihood.")
    st.write("**Recommendation:** Focus on monitoring and managing blood pressure, especially in older patients, to reduce stroke risk.")

else:
    st.write("Please upload a CSV file.")
