# Fintech Loan Prediction

## “Predicting Personal Loan Risk Level by Analyzing Borrower Financial History, Credit Worthiness and Desired Loan Configurations”

**Industry**: Finance / Fintech

---

## Problem Statement

In today's fast-paced lending market, competition is fierce. Emerging fintech platforms, P2P lending, and DeFi have lowered entry barriers for non-traditional lenders. To stay competitive, lenders must vet, price, and approve loans rapidly and accurately.

Traditional lenders are slowing down due to increased regulation, opening up opportunities for agile fintechs. A robust predictive model to assess loan quality and default risk is essential for managing risk exposure and gaining a competitive edge.

---

## Project Goals

- Streamline approval processes for low-risk loans
- Assign fair pricing tiers (interest rates) based on predicted risk
- Flag and monitor high-risk loans for portfolio management

---

## Process Overview

- Use borrower financial data (income, credit score, DTI, defaults, etc.)
- Analyze loan details (term, amount, purpose)
- Perform visual exploration to uncover key trends
- Build a predictive model to assign loan risk level
- Optionally, flag already-issued loans with high default risk for monitoring

---

## Objectives

- Clean and analyze real-world Lending Club 2018 loan data
- Visualize key financial behaviors and credit trends
- Train and optimize classification models (Random Forest, XGBoost, Neural Networks)
- Handle class imbalance using SMOTE and class weighting
- Track model performance with precision, recall, F1-score, and confusion matrix

---

## Dataset

**Source**: Lending Club (via Kaggle)  
**Sample size**: ~1 million personal loan records  
**Features**: income, loan amount, DTI, employment status, loan purpose, loan grade, and more

---

## Project Steps

### 1. Data Cleaning
- Reduced original 145 columns to 27 relevant features
- Removed null values and duplicates
- Handled outliers (e.g., income > $1M)

### 2. Exploratory Data Analysis (EDA)
- Visualized distributions of income, loan amount, DTI, interest rate, loan grades
- Used heatmaps, boxplots, and bar plots to identify correlations
- Confirmed class imbalance (fewer high-risk loans)
- Final cleaned dataset saved as `df_subset.csv`

### 3. Database Schema
- Created schema in PostgreSQL
- Uploaded cleaned data to AWS RDS

### 4. Data Preprocessing
- Encoded categorical variables using `get_dummies()`
- Bucketed values to smooth distributions
- Scaled numeric values with `StandardScaler`

### 5. Modeling
- Models Used:
  - Random Forest Classifier
  - XGBoost Classifier
  - Neural Network (Keras)
- Target: `loan_grade_grouped`  
  (Low Risk = Grades A/B, High Risk = Grades C–G)
- Split: 80/20 training/test
- Best F1 Score: ~75%
- Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix

### 6. Optimization & Class Imbalance Handling
- Handled imbalance using:
  - SMOTE
  - Class weighting
- Hyperparameter tuning:
  - Tree depth, estimators, activation functions
  - Epochs, batch size (NN)
- Incorporated `SimpleImputer` and consistent scaling

### 7. Model Outputs
- Loan risk prediction (Low vs High)
- Confusion matrix, classification report

---

## Technologies Used

- Python, Jupyter, Google Colab
- pandas, numpy for data processing
- matplotlib, seaborn, plotly for data visualization
- scikit-learn, xgboost, keras for machine learning and deep learning
- imbalanced-learn for handling class imbalance with SMOTE
- joblib and SQLAlchemy for model saving and database integration
- Gradio for interactive ML interface
- Flask for lightweight backend deployment
- PostgreSQL on AWS RDS for relational database storage
- AWS for cloud hosting and database services

---

## Repository Structure

Repo Schema Project4-Loan_Analysis/

│ ├── 📁 data/ # Raw and cleaned data files

│ ├── loan_2018 - Loan data for 2018 only used in intial EDA

│ └── full_loan_data.csv - full set of data from Lending Club (US-based lending platform)

| └── df_subset - subset of the full_loan_data.csv that was used for the analysis of potential models and the creation of the AWS cloud data set

| └── LCDataDictionary - explains the columns/headers for the full_loan_data file

│ ├── 📁 notebooks/ # Jupyter Notebooks (EDA, model training, etc.)

| ├── EDA_Loan_Data.ipynb

| └── data_cleaning

| 📁 all_other_models/

     └── A compilation of m compilation of multiple model attempts used to find the best combination of model type, hyperams, columns, etc. to arrive at the final model
---

## Next Steps (Future Work)

- Use a larger dataset and advanced balancing techniques
- Host full web app with Streamlit or Flask
- Integrate real-time credit bureau API (e.g., Equifax) for live scoring

---

## License

Project 4 – Data Analytics Boot Camp – University of Toronto  
For educational purposes only

---

## Team

- Seyhr Waqas  
- Asif Shahzad  
- José Traboulsi  
- Karla Lopez Marin
