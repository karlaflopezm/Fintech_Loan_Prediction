import gradio as gr
import pandas as pd
import joblib

# Load model and features
model = joblib.load("models/rf_model_tuned.pkl")
features = joblib.load("models/rf_model_features_tunned.pkl")

# Prediction function
def predict_risk(term, home_ownership, annual_inc, loan_amount, dti_level,
                 emp_length, purpose):
    
    # DTI mapping
    dti_mapping = {
        "Low (<15%)": 12.0,
        "Medium (15–30%)": 22.5,
        "High (30–45%)": 37.5
    }
    dti = dti_mapping[dti_level]

    # Prepare input
    input_data = {
        "term": term,
        "home_ownership": home_ownership,
        "merged_annual_inc": annual_inc,
        "funded_amnt": loan_amount,
        "merged_dti": dti,
        "emp_length_grouped": emp_length,
        "merged_purpose": purpose
    }

    # One-hot encode and align with training features
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)

    # Predict
    pred = model.predict(df)[0]
    return "Low Risk" if pred == 0 else "High Risk"

# Gradio inputs 
inputs = [
    gr.Number(label="Loan Amount", value=10000),
    gr.Dropdown([36, 60], label="Loan Term", value=36),
    gr.Dropdown(["RENT", "MORTGAGE", "OWN", "OTHER"], label="Home Ownership", value="RENT"),
    gr.Number(label="Annual Income", value=55000),
    gr.Dropdown(["Low (<15%)", "Medium (15–30%)", "High (30–45%)"],
                label="Debt-to-Income Ratio (DTI)", value="Medium (15–30%)"),
    gr.Dropdown(["< 1 year", "1-3 years", "3-5 years", "5-10 years", "10+ years"],
                label="Employment Length", value="5-10 years"),
    gr.Dropdown(["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "medical", "small_business", "other"],
                label="Loan Purpose", value="debt_consolidation")
]

#  Launch app
gr.Interface(
    fn=predict_risk,
    inputs=inputs,
    outputs="text",
    title="Loan Risk Classifier",
    description="Enter borrower information to predict if a loan is High or Low Risk using a Random Forest model."
).launch()
