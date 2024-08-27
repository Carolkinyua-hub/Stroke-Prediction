# Stroke Prediction App

The Stroke Prediction App is a Streamlit-based web application designed to predict stroke occurrences using a pre-trained neural network model. This app allows users to upload their own datasets, process the data, and visualize various performance metrics and feature distributions.

## Features

- **Upload CSV File**: Users can upload a CSV file containing stroke-related data.
- **Data Processing**: 
  - Data loading into a DataFrame.
  - Downsampling of the majority class to balance the dataset.
  - Shuffling of the balanced dataset.
- **Feature and Target Separation**: Separates features and the target variable (Stroke) in the balanced dataset.
- **Data Scaling**: Scales features using MinMaxScaler to ensure uniformity.
- **Model Prediction**:
  - Loads a pre-trained neural network model.
  - Makes predictions on the scaled features.
- **Evaluation and Visualization**:
  - Displays stroke prevalence.
- **Feature-wise Analysis**:
  - Generates feature importance.

## Installation

To run this app locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stroke-prediction-app.git
