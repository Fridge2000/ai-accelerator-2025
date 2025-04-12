import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# Define the same model class
class MultiLabelNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLabelNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load column names (assumes they match what you used in Colab)
df = pd.read_csv('encoded_dataset.csv')  # Place the same CSV in your project directory
disease_columns = [col for col in df.columns if "Disease_" in col]
symptom_columns = [col for col in df.columns if col not in disease_columns]
df["Disease"] = df[disease_columns].idxmax(axis=1).str.replace("Disease_", "")
df = df.drop(columns=disease_columns)
df[symptom_columns] = df[symptom_columns].astype(int)

# Model setup
input_size = len(symptom_columns)
hidden_size = 128
output_size = df["Disease"].nunique()

model = MultiLabelNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("thirdopinionmodel.pth", map_location=torch.device('cpu')))
model.eval()

# Get disease labels
disease_labels = sorted(df["Disease"].unique())

# Streamlit interface
st.title("Disease Prediction App")

st.write("Please select your symptoms:")

selected_symptoms = []
for symptom in symptom_columns:
    if st.checkbox(symptom.replace("_", " ").capitalize()):
        selected_symptoms.append(symptom)

# Prediction on form submission
if st.button("Predict Disease"):
    input_vector = np.zeros(len(symptom_columns))
    for i, symptom in enumerate(symptom_columns):
        if symptom in selected_symptoms:
            input_vector[i] = 1

    if not any(input_vector):
        st.warning("Please select at least one symptom.")
    else:
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            outputs = torch.sigmoid(model(input_tensor))
            st.write("Raw prediction scores:", outputs.numpy())  # Debugging
            predictions = (outputs > 0.02).int().squeeze().tolist()

        st.subheader("Prediction Results:")
        any_positive = False
        for i, pred in enumerate(predictions):
            if pred:
                st.write(f"- {disease_labels[i]}")
                any_positive = True

        if not any_positive:
            st.write("No diseases predicted based on the selected symptoms.")
