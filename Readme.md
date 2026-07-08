# Loan Approval Prediction Case Study

## 📌 Project Overview
The objective of this project is to build a robust supervised machine learning model to predict whether a borrower's loan application should be approved or rejected. This project emphasizes end-to-end machine learning practices, with a strong focus on data preprocessing, mitigating class imbalance, and providing business-oriented evaluation metrics.

## 📊 Dataset
**Source:** [Kaggle - Loan Approval Prediction Case Study](https://www.kaggle.com/datasets/bhanupratapbiswas/loan-approval-prediction-case-study)  
**File:** `loan_prediction.csv`

The initial dataset exhibited a significant class imbalance in the target variable (`Loan_Status`), with approvals constituting 68.7% of the data and rejections making up 31.3%.

## 🛠️ Methodology & Pipeline

### 1. Data Preprocessing
* **Feature Selection:** The `Loan_ID` column was dropped as it holds no predictive power for the model.
* **Handling Missing Values:** Missing numerical features (`LoanAmount`, `Loan_Amount_Term`, `Dependents`) were imputed using the median to avoid outlier skew, while missing categorical features (`Gender`, `Married`, `Self_Employed`, `Credit_History`) were filled using the mode.
* **Feature Engineering:** The `Dependents` column string values of '3+' were mapped to '3' and converted to float types for mathematical processing. 
* **Encoding:** Categorical variables were converted to binary formats using One-Hot Encoding (`get_dummies` with `drop_first=True`), and the target variable was mapped to binary integers (Y=1, N=0).
* **Scaling:** `StandardScaler` was applied to normalize the feature distributions, which is highly critical for distance-based models and Logistic Regression.

### 2. Handling Class Imbalance
To prevent the model from becoming biased toward the majority class (Approved loans), **SMOTE (Synthetic Minority Over-sampling Technique)** was applied strictly to the training data.
* **Before SMOTE:** 337 Approved (1) vs. 154 Rejected (0).
* **After SMOTE:** The classes were perfectly balanced at 337 instances each.

## 📈 Model Performance & Evaluation

Two models were trained and evaluated on an 80/20 stratified split: Logistic Regression and Random Forest. Logistic Regression outperformed Random Forest across almost all key metrics, making it the superior choice for this specific dataset.

| Metric | Logistic Regression | Random Forest |
| :--- | :--- | :--- |
| **Accuracy** | 0.81 | 0.80 |
| **Precision (Class 1)** | 0.85 | 0.83 |
| **Recall (Class 1)** | 0.88 | 0.88 |
| **F1-Score (Class 1)** | 0.87 | 0.86 |
| **ROC-AUC Score** | **0.8703** | 0.7932 |

### Confusion Matrix Insights (Logistic Regression)
Based on the test set, the Logistic Regression model yielded the following matrix:
* **True Positives (TP):** 75 (Correctly approved)
* **True Negatives (TN):** 25 (Correctly rejected)
* **False Positives (FP):** 13 (Incorrectly approved)
* **False Negatives (FN):** 10 (Incorrectly rejected)

## 💼 Business Interpretation & Deployment Strategy

In the context of loan approvals, the two types of model errors carry drastically different business costs:

1. **The Cost of False Positives (Type I Error):** The model approved 13 loans that should have been rejected. In a real-world banking scenario, this is the most dangerous error. Approving a bad loan leads to financial defaults, unrecoverable capital, and significant institutional loss.
2. **The Cost of False Negatives (Type II Error):** The model rejected 10 loans that would have been successfully paid back. The business cost here is lost interest revenue and potential customer dissatisfaction.

### Suggested Deployment Threshold
By default, Logistic Regression uses a 0.5 probability threshold to classify approvals. Given that the financial risk of a default (False Positive) usually outweighs the cost of lost interest (False Negative), **the deployment threshold should be adjusted upward.**

Moving the decision threshold from 0.5 to a stricter level (e.g., 0.60 or 0.65) will force the model to be more confident before approving a loan. This trade-off will naturally decrease False Positives (saving the bank from defaults) while marginally increasing False Negatives, resulting in a safer, more conservative lending strategy. 

---
### Author
**N Tharun** B.Tech Data Science, SITAMS  
*Contact:* tharunreddy1516@gmail.com
