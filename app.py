import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the pre-trained model
model_path = 'path_to_your_pretrained_model.joblib'
scaler_path = 'path_to_your_scaler.joblib'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def preprocess_data(df):
    # Ensure you have the required columns
    required_columns = ['HeartDiseaseorAttack', 'HighBP', 'BMI', 'GenHlth', 'MentHlth',
                        'PhysHlth', 'DiffWalk', 'Age', 'Education', 'Income']
    df = df[required_columns]
    df_scaled = scaler.transform(df)
    return df_scaled

def predict(model, data):
    return model.predict_proba(data)[:, 1]

def visualize_data(df):
    st.subheader("Prevalence vs Features")

    features = ['HeartDiseaseorAttack', 'HighBP', 'BMI', 'GenHlth', 'MentHlth',
                'PhysHlth', 'DiffWalk', 'Age', 'Education', 'Income']
    
    for feature in features:
        st.subheader(f"{feature} vs Stroke Prevalence")
        fig, ax = plt.subplots()
        df[feature].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f"{feature} Distribution")
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        st.pyplot(fig)

def main():
    st.title("Stroke Prediction App")

    # Data Upload
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(df.head())

        # Data Preprocessing
        st.subheader("Data Preprocessing")
        try:
            df_scaled = preprocess_data(df)
            st.write("Preprocessed Data:")
            st.write(df_scaled[:5])
        except KeyError as e:
            st.error(f"Error in preprocessing: Missing column {e}")
            return

        # Model Prediction
        st.subheader("Model Prediction")
        predictions = predict(model, df_scaled)
        df['Stroke Probability'] = predictions
        st.write("Prediction Results:")
        st.write(df[['Stroke Probability']].head())

        # Visualization
        st.subheader("Visualizations")
        visualize_data(df)

if __name__ == "__main__":
    main()
