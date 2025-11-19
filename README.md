# Give Me Some Credit -- Default Prediction Model

This project builds a complete machine learning pipeline to predict
whether a customer will experience financial distress in the next two
years using the **Give Me Some Credit** dataset.

The goal is to create a **clean, interpretable, and high-performing
model** using classical ML algorithms such as Random Forest and Logistic
Regression.

------------------------------------------------------------------------

## Final Results Summary

### **Model Performance (Validation Set)**

  Model                     AUC         Precision   Recall   F1-score
  ------------------------- ----------- ----------- -------- ----------
  **Random Forest**         **0.867**   0.255       0.199    0.224
  **Logistic Regression**   **0.859**   0.239       0.160    0.192

### Interpretation

-   **AUC 0.867 (Random Forest)** → This is considered **strong
    performance** for an imbalanced binary classification problem.
-   Precision/recall scores are low because:
    -   Only \~7% of the samples are positive.
    -   Classical ML models struggle with rare events.
    -   The priority metric for this dataset is **AUC**, not accuracy.

**Random Forest outperforms Logistic Regression**.

------------------------------------------------------------------------

## 📂 Project Structure

    ├── cs-training.csv
    ├── cs-test.csv
    ├── submission.csv
    ├── train_notebook.ipynb
    └── README.md

------------------------------------------------------------------------

## Data Processing Pipeline

### **1️ Load Dataset**

-   Training set contains `SeriousDlqin2yrs`
-   Test set does not contain the target

### **2️ Feature Engineering**

Applied to both training and test (safe since test has no target):

-   Missing value imputation\
-   Log-scaling of skewed financial variables\
-   Derived features such as:
    -   TotalPastDue
    -   DebtRatio adjustments
    -   Combined delinquency features

### **3️ Train--Validation Split**

20% validation for model evaluation.

------------------------------------------------------------------------

## Models Used

### **1. Random Forest Classifier**

-   Handles nonlinear relationships
-   Robust to outliers
-   Provides feature importance
-   Best-performing model

### **2. Logistic Regression**

-   Linear baseline model
-   Scaled using StandardScaler
-   Used for comparison

------------------------------------------------------------------------

## Feature Importance (Random Forest)

Top predictors of financial distress:

1.  RevolvingUtilizationOfUnsecuredLines
2.  TotalPastDue
3.  PastDueTwo
4.  NumberOfTimes30-59DaysPastDueNotWorse
5.  Age
6.  NumberOfTimes90DaysLate
7.  DebtRatio
8.  NumberOfOpenCreditLines
9.  NumberRealEstateLoansOrLines
10. MonthlyIncome

These features align with real-world credit risk indicators.

------------------------------------------------------------------------

## Metrics Used

Evaluation metrics include:

-   **AUC ROC** (primary metric)
-   Precision
-   Recall
-   F1-score
-   Feature importance plots

AUC is the most reliable metric given the **highly imbalanced** nature
of the dataset.

------------------------------------------------------------------------

##  Submission

-   Predictions generated using the Random Forest model
-   Test data columns aligned with training columns
-   Output saved to **submission.csv**

------------------------------------------------------------------------

##  How to Run

``` python
train_raw, test_raw = load_data(TRAIN_PATH, TEST_PATH)

train = feature_engineering(train_raw)
test  = feature_engineering(test_raw)

X_train, X_val, y_train, y_val, X_all = create_splits(train)

rf_model, auc_rf = train_random_forest(X_train, y_train, X_val, y_val)
lr_model, auc_lr, scaler = train_logistic_regression(X_train, y_train, X_val, y_val)

# predictions + evaluation + submission steps follow
```

------------------------------------------------------------------------

##  Conclusion

-   Achieved **AUC 0.867** on validation using Random Forest\
-   Strong predictive performance for a highly imbalanced financial
    dataset\
-   Clean pipeline, interpretable features, and reproducible results\
-   Suitable for risk ranking and credit scoring use cases

------------------------------------------------------------------------

##  Future Improvements

-   Use **XGBoost** or **LightGBM**
-   Hyperparameter tuning (Optuna / GridSearch)
-   Class-balancing techniques (SMOTE, class weights)
-   Cost-sensitive learning for higher recall
-   Model stacking/ensembling

------------------------------------------------------------------------

##  Author

Machine Learning Research & Implementation\
Built with Python, Scikit-Learn, Pandas, Matplotlib
