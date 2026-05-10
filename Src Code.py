# =========================================================
# TELECOM CUSTOMER CHURN PREDICTION PROJECT
# FINAL PROFESSIONAL VERSION
# =========================================================


# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


# =========================================================
# 2. LOAD DATASET
# =========================================================

df = pd.read_csv("../data/telecom_churn.csv")


# =========================================================
# 3. DISPLAY DATASET
# =========================================================

print("First 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# =========================================================
# 4. DATASET INFORMATION
# =========================================================

print("\nDataset Information:\n")
print(df.info())


# =========================================================
# 5. CHECK MISSING VALUES
# =========================================================

print("\nMissing Values:\n")
print(df.isnull().sum())


# =========================================================
# 6. STATISTICAL SUMMARY
# =========================================================

print("\nStatistical Summary:\n")
print(df.describe())


# =========================================================
# 7. CHECK TARGET VARIABLE
# =========================================================

print("\nChurn Values:")
print(df["churn"].unique())


# =========================================================
# 8. DROP UNNECESSARY COLUMN
# =========================================================

df.drop(
    "customer_id",
    axis=1,
    inplace=True
)


# =========================================================
# 9. CONVERT DATE COLUMN
# =========================================================

df["date_of_registration"] = pd.to_datetime(
    df["date_of_registration"],
    dayfirst=True
)


# =========================================================
# 10. FEATURE ENGINEERING FROM DATE
# =========================================================

df["registration_year"] = (
    df["date_of_registration"].dt.year
)

df["registration_month"] = (
    df["date_of_registration"].dt.month
)


# =========================================================
# 11. DROP ORIGINAL DATE COLUMN
# =========================================================

df.drop(
    "date_of_registration",
    axis=1,
    inplace=True
)


# =========================================================
# 12. ENCODE CATEGORICAL COLUMNS
# =========================================================

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

print("\nCategorical Columns:")
print(categorical_columns)

le = LabelEncoder()

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])


# =========================================================
# 13. FINAL DATASET CHECK
# =========================================================

print("\nProcessed Dataset:\n")
print(df.head())


# =========================================================
# ====================== EDA ==============================
# =========================================================


# =========================================================
# 14. CHURN DISTRIBUTION
# =========================================================

plt.figure(figsize=(6,4))

sns.countplot(
    x="churn",
    data=df
)

plt.title("Customer Churn Distribution")

plt.show()


# =========================================================
# 15. CORRELATION HEATMAP
# =========================================================

plt.figure(figsize=(14,10))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()


# =========================================================
# 16. AGE VS CHURN
# =========================================================

plt.figure(figsize=(6,4))

sns.boxplot(
    x="churn",
    y="age",
    data=df
)

plt.title("Age vs Churn")

plt.show()


# =========================================================
# 17. ESTIMATED SALARY VS CHURN
# =========================================================

plt.figure(figsize=(6,4))

sns.boxplot(
    x="churn",
    y="estimated_salary",
    data=df
)

plt.title("Estimated Salary vs Churn")

plt.show()


# =========================================================
# 18. DATA USAGE VS CHURN
# =========================================================

plt.figure(figsize=(6,4))

sns.boxplot(
    x="churn",
    y="data_used",
    data=df
)

plt.title("Data Usage vs Churn")

plt.show()


# =========================================================
# ================= MACHINE LEARNING ======================
# =========================================================


# =========================================================
# 19. FEATURES AND TARGET
# =========================================================

X = df.drop("churn", axis=1)

y = df["churn"]


# =========================================================
# 20. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# 21. FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =========================================================
# 22. APPLY SMOTE
# =========================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())


# =========================================================
# 23. XGBOOST MODEL
# =========================================================

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric="logloss"
)


# =========================================================
# 24. TRAIN MODEL
# =========================================================

xgb.fit(
    X_train_smote,
    y_train_smote
)


# =========================================================
# 25. PREDICTIONS
# =========================================================

y_pred = xgb.predict(X_test_scaled)


# =========================================================
# 26. ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(accuracy)


# =========================================================
# 27. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# =========================================================
# 28. ROC AUC SCORE
# =========================================================

roc_score = roc_auc_score(
    y_test,
    y_pred
)

print("\nROC-AUC Score:")
print(roc_score)


# =========================================================
# 29. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# =========================================================
# ================= FEATURE IMPORTANCE ====================
# =========================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:\n")
print(feature_importance)


# =========================================================
# 30. FEATURE IMPORTANCE PLOT
# =========================================================

plt.figure(figsize=(10,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance
)

plt.title("Feature Importance")

plt.show()


# =========================================================
# 31. ROC CURVE
# =========================================================

y_prob = xgb.predict_proba(
    X_test_scaled
)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.show()


# =========================================================
# 32. FINAL CONCLUSION
# =========================================================

print("\nProject Completed Successfully.")
print("Final Model Used: XGBoost Classifier")