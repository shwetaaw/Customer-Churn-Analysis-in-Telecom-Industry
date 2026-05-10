# Telecom Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn in the telecom industry using Machine Learning techniques. Customer churn prediction helps telecom companies identify customers who are likely to leave the service, allowing businesses to take proactive retention measures.

The project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Handling imbalanced data using SMOTE
- Machine Learning model building using XGBoost
- Model evaluation and visualization

---

## Problem Statement

Customer churn is one of the major challenges faced by telecom companies. The objective of this project is to build a machine learning model that predicts whether a customer is likely to churn based on demographic information, service usage patterns, and customer-related attributes.

---

## Dataset Information

The dataset contains telecom customer information such as:
- Telecom partner
- Gender
- Age
- State and city
- Number of dependents
- Estimated salary
- Calls made
- SMS sent
- Data usage
- Registration details
- Churn status

### Target Variable
- `0` → Customer retained
- `1` → Customer churned

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost

---

## Project Workflow

1. Data Collection and Loading  
2. Data Cleaning and Preprocessing  
3. Feature Engineering  
4. Exploratory Data Analysis  
5. Handling Imbalanced Data using SMOTE  
6. Feature Scaling  
7. Model Training using XGBoost  
8. Model Evaluation  
9. Feature Importance Analysis  
10. ROC Curve Visualization  

---

## Exploratory Data Analysis

The project includes multiple visualizations such as:
- Churn distribution
- Correlation heatmap
- Age vs churn analysis
- Salary vs churn analysis
- Data usage vs churn analysis
- Feature importance plot
- ROC curve
- Confusion matrix

---

## Machine Learning Model

### Final Model Used
- XGBoost Classifier

### Evaluation Metrics
- Accuracy Score
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## Results

The model achieved acceptable overall accuracy for customer retention prediction. However, the model showed limited capability in identifying churn customers due to weak feature relationships and class imbalance in the dataset.

Key observations:
- City and number of dependents were among the most influential features.
- Telecom partner and registration period also contributed to churn prediction.
- Additional telecom behavioral features could further improve model performance.

---

## Feature Importance

Top influential features:
1. City
2. Number of Dependents
3. Telecom Partner
4. Registration Month
5. Registration Year

---

## Future Improvements

Possible future enhancements:
- Hyperparameter tuning
- Advanced ensemble methods
- Inclusion of customer complaint history
- Recharge frequency analysis
- Customer service interaction data
- Real-world telecom behavioral features

---

## Project Structure

telecom-customer-churn-prediction/
│
├── data/
│   └── telecom_churn.csv
│
├── notebooks/
│   └── churn_prediction.ipynb
│
├── requirements.txt
├── README.md

---

## Installation

Clone the repository:

```bash
git clone <your-github-repository-link>