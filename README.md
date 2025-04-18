# Fintech_Loan_Prediction

## “Predicting Personal Loan Risk Level by Analyzing Borrower Financial History, Credit Worthiness and Desired Loan Configurations”

## Industry: Finance / Fintech

### Problem
In today's fast paced lending markets, competition for market share is fierce. Technological developments like peer-to-peer lending, apps, DeFi, have lowered entry barriers for new lenders to enter the market. As banks / traditional lenders retrench (and traditional credit sources dry up) given increased regulatory pressure, opportunities emerge, but non-traditional players must act fast and offer competitive solutions in a timely manner to maintain market share and increase brand awareness. This means that non-traditional lenders able to vet, price, and approve loans quickly will have a competitive edge. As such, having a model capable of predicting loan quality / default risk is paramount to a non-traditional lending firm to stay relevant and manage risk exposure.

### Process Overview

- Utilize borrower financial data (e.g., income, credit history, debt-to-income ratio, defaults) and loan-specific details (e.g., term, purpose).
- Perform visual exploration to uncover key trends and risk indicators.
- Build a predictive classification model to assess **loan risk** and assist in **interest rate assignment** or **loan approval decisions**.
- *(Bonus)* Extend the model to flag issued loans that are at high risk of default — useful for **portfolio monitoring** and **proactive risk management**.

## Project Goals

- Streamline approval processes for **low-risk loans**
- Assign **fair pricing tiers (interest rates)** based on risk
- Monitor and manage **high-risk loans** more cautiously

## Objectives

- Clean and analyze real-world loan data (Lending Club 2018 dataset)
- Visualize financial behaviors and credit trends
- Train and test a predictive model (Random Forest, XGBoost, Neural Networks)
- Handle class imbalance using techniques like SMOTE and class weighting
- Iteratively optimize the model and evaluate performance using precision, recall, F1-score, and confusion matrix

 **Dataset**

Lending Club Loan Dataset (Kaggle)
Sample size: ~1 million personal loans
Features: income, loan amount, DTI, employment, loan purpose, loan grade, and more

### Project Steps

## 1. Data Cleaning
- Reduced 145 original columns to **27 relevant features** based on EDA and business logic
- Removed **null values** and **duplicate records**
- Handled outliers (e.g., removed income values > $1M)

## 2. Exploratory Data Analysis (EDA)
- Visualized distributions:
  - Annual income
  - Debt-to-Income 
  - (DTI) ratio
  - Loan amount
  - Interest rates
  - Grade categories
- Identified feature relationships with loan grade using:
  - Correlation heatmaps
  - Boxplots
  - Bar plots
- Confirmed **class imbalance**: significantly fewer high-risk (F/G) loans
- Final cleaned dataset saved as `df_subset.csv`

## 3. Database Schema
- Created a PostgreSQL schema and uploaded cleaned data to **AWS RDS** for long-term accessibility and scalability.

## 4. Data Preprocessing
- Encoded **categorical variables** using `get_dummies()` (e.g., loan purpose, home ownership)
- Bucketed categorical and numerical features to smooth distributions
- Applied **StandardScaler** to normalize numerical variables


## 5. Modeling
- Tested three machine learning models:
  - Random Forest Classifier
  - Neural Network (Keras Sequential API)
  - XGBoost Classifier
- Target variable:
  - `loan_grade_grouped` → Low Risk (A/B) vs High Risk (C–G)
- Training/Test split: **80/20**
- Achieved up to **75% F1 Score**
- Evaluated using:
  - Accuracy, Precision, Recall
  - Confusion Matrix

## 6. Optimization & Class Imbalance Handling
- Addressed **class imbalance** using:
  - **SMOTE** (Synthetic Minority Over-sampling)
  - **Class weights** in models
- Hyperparameter tuning strategies:
  - Tested different tree depths, estimators, and activation functions
  - Tuned batch size and epochs for neural networks
  - Compared results across feature bucket versions
- Incorporated:
  - **SimpleImputer** for missing values
  - **StandardScaler** consistently across models
- Chose model based on best balance of **recall and precision**, with special focus on catching **Grade C–G (high risk)** loans

## 7.Model Outputs
- Outputs Loan risk prediction (Low vs High Risk)
- Confusion matrix, classification report


## Technologies Used
### Core Technologies
- **Python**: Core scripting language for development.
- **Jupyter / Colab**: Interactive environments for analysis and development.
### Data Manipulation and Analysis
- **Pandas**: Data cleaning and transformation.
- **NumPy**: Numerical computations and array manipulation.
### Data Visualization
- **Matplotlib**: Fundamental data visualization.
- **Seaborn**: Statistical plotting.
- **Plotly**: Interactive charts and graphs.
### Machine Learning
- **scikit-learn**: Machine learning modeling and evaluation.
- **XGBoost**: High-performance gradient boosting.
- **Keras (TensorFlow backend)**: Deep learning framework.
- **imbalanced-learn (SMOTE)**: Addressing class imbalances in datasets.
### Backend Development
- **Gradio**: User-friendly interfaces for ML models.
- **Flask**: Lightweight web framework for deployment.
### Database and Cloud Services
- **SQL / SQLAlchemy**: Database queries and object-relational mapping (ORM).
- **PostgreSQL (AWS RDS)**: Relational database hosted on AWS.
- **AWS**: Cloud infrastructure for hosting and computing.


**Repo Schema**
Project4-Loan_Analysis/

│
├── 📁 data/                        # Raw and cleaned data files

│   ├── loan_2018 - Loan data for 2018 only used in intial EDA

│   └── full_loan_data.csv - full set of data from Lending Club (US-based lending platform)

|   └── df_subset - subset of the full_loan_data.csv that was used for the analysis of potential models and the creation of the AWS cloud data set

|   └── LCDataDictionary - explains the columns/headers for the full_loan_data file


│
├── 📁 notebooks/                  # Jupyter Notebooks (EDA, model training, etc.)

|   ├── EDA_Loan_Data.ipynb

|   └── data_cleaning

|   📁 all_other_models/  

         └── A compilation of multiple model attempts used to find the best combination of model type, hyperams, columns, etc. to arrive at the final model

├── 📁 final/                    # notebooks/scripts for final model

│   ├── db_config.json

|   └── database_creation.ipynb

|   └── randomforest.ipynb
    
├── 📁 visuals/                    # Graphs, model performance charts, etc. (FINALIZE)

│   ├── feature_importance.png

│   └── confusion_matrix.png

├── 📁 docs/                       # Presentation slides, summary PDF, etc.

├── README.md                     # Final project README


**Next Steps (Optional Future Work)**
Using a larger data set and more advanced sampling/balancing techniques to better handle imbalanced classes and improve F1 scores
Build a web app using Streamlit or Flask for interactive use
Integrate real-time credit bureau API (e.g., Equifax) for feature enrichment

 **License**
Project 4 – Data Analytics Boot Camp – University of Toronto
For educational purposes only.

 **Team**
Asif Shahzad
José Traboulsi
Seyhr Waqas
Karla Lopez Marin



