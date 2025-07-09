import gradio as gr
import pickle
import numpy as np

# Load your trained model
with open("diabetes_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define prediction function
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness,
                     insulin, bmi, diabetes_pedigree_function, age):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree_function, age]])
    prediction = model.predict(input_data)
    return "🟢 Not Diabetic" if prediction[0] == 0 else "🔴 Diabetic"

# Gradio interface
iface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age")
    ],
    outputs="text",
    title="Diabetes Prediction App",
    description="Enter health parameters to check diabetes risk using a trained ML model."
)

iface.launch()
