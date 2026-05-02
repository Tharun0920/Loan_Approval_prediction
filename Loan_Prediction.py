#Import neccesary Libraries 
import imblearn
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve


if __name__ == "__main__":
    df = pd.read_csv('loan_prediction.csv')

    # 1. Drop Loan_ID as it has no predictive power
    df = df.drop('Loan_ID', axis=1)

    # 2. Convert Target Variable to Binary (Y=1, N=0)
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # 3. Handle 'Dependents' column (contains '3+' string)
    df['Dependents'] = df['Dependents'].replace('3+', '3').astype(float)

    # 4. Fill Missing Values
    # Numerical missing values -> Median
    num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Dependents']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical missing values -> Mode (including Credit_History which acts categorically)
    cat_cols = ['Gender', 'Married', 'Self_Employed', 'Credit_History']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 5. One-Hot Encode Categorical Variables
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'], drop_first=True)

    print("Target Class Distribution:\n", df['Loan_Status'].value_counts(normalize=True))
    print(df.head())

    # Separate features and target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature Scaling (Important for models like Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE to the training data to handle the 2:1 class imbalance
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    print(f"Before SMOTE: \n{y_train.value_counts()}")
    print(f"After SMOTE: \n{y_train_smote.value_counts()}")

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
    }

    # Train and evaluate
    for name, model in models.items():
        print(f"--- {name} ---")
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}\n")
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'{name} - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
