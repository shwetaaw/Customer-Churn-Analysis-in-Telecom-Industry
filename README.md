# Customer Churn Analysis in Telecom Industry

This repository contains a mini data science project for the **Fundamentals of Data Science – Credit Activity 2** (M.Sc. Data Science, Sem-I, Dr. D. Y. Patil ACS College, Pimpri).

The objective of this project is to **study and analyse customer churn data in the telecom industry**, identify key factors that lead to churn, and build a simple predictive model.

---

## 1. Problem Statement

Telecom companies lose a significant amount of revenue when customers discontinue their services (churn). The goal of this project is to:

- Analyse customer data.
- Understand patterns and factors associated with churn.
- Build a basic model to predict whether a customer is likely to churn.

---

## 2. Dataset

- **Source:** Any public telecom churn dataset (e.g., Kaggle / UCI).  
- **File used in this project:** `data/telecom_churn.csv`  
- **Example columns:**
  - `customerID`
  - `gender`
  - `SeniorCitizen`
  - `Partner`
  - `Dependents`
  - `tenure`
  - `PhoneService`
  - `MultipleLines`
  - `InternetService`
  - `Contract`
  - `PaymentMethod`
  - `MonthlyCharges`
  - `TotalCharges`
  - `Churn` (Yes/No)

> Note: You can replace the dataset with any similar customer churn dataset, just ensure that a `Churn` column and basic customer features are present.

---

## 3. Project Structure

```text
telecom-churn-analysis/
├─ data/
│  └─ telecom_churn.csv        # dataset file (not included here)
├─ src/
│  └─ churn_analysis.py        # main analysis and modeling script
├─ notebooks/                  # optional Jupyter notebooks
├─ README.md                   # project documentation
├─ requirements.txt            # Python dependencies
└─ .gitignore                  # files to ignore in Git
