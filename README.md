# Telco Customer Churn Prediction

## Project Overview

Customer churn is a major business challenge for telecommunications companies because losing existing customers can directly affect recurring revenue.

This project develops an end-to-end machine learning solution to predict whether a telecom customer is likely to churn based on customer demographics, services, contract details, payment method, tenure, and billing information.

The project covers the complete data science workflow, from data cleaning and exploratory data analysis to model development, evaluation, model deployment, and a Streamlit-based prediction application.

## Business Problem

The objective is to identify customers who are at higher risk of churn so that the business can better understand churn patterns and potentially prioritize customers for retention activities.

The target variable is:

* **Churn = Yes** → Customer churned
* **Churn = No** → Customer did not churn

This is formulated as a **binary classification problem**.


## Dataset

The project uses the IBM Telco Customer Churn dataset.

* **Rows:** 7,043
* **Original columns:** 21
* **Target variable:** `Churn`
* **Churn = Yes:** 1,869 customers
* **Churn = No:** 5,174 customers

The dataset contains customer information across demographics, account details, subscribed services, contract type, payment method, and billing information.

### Main Feature Groups

* Customer demographics
* Customer tenure
* Phone and internet services
* Additional subscribed services
* Contract information
* Billing information
* Payment method

The `customerID` column was excluded from model training because it is an identifier rather than a predictive customer attribute.

## Project Objectives

1. Understand the factors associated with customer churn.
2. Clean and prepare the raw customer data.
3. Perform exploratory data analysis to identify important patterns.
4. Build and compare multiple classification models.
5. Evaluate models using appropriate classification metrics.
6. Select a final model based on validation performance and the project objective.
7. Determine an appropriate classification threshold using training data.
8. Save the trained model for reuse.
9. Build a Streamlit application for customer-level churn prediction.


## Data Preparation

The raw dataset was inspected for data types, missing values, duplicate records, and invalid values.

### Key Data Cleaning Steps

* Identified 11 blank values in `TotalCharges`.
* Converted `TotalCharges` from object to numeric.
* The 11 blank records belonged to customers with `tenure = 0`, so they were treated as `TotalCharges = 0` based on the customer billing context.
* Verified that there were no duplicate records or remaining missing values.
* Removed `customerID` from model training because it is an identifier rather than a predictive feature.

### Feature Preparation

* Separated the target variable `Churn` from the input features.
* Applied `OneHotEncoder` to categorical features.
* Applied `StandardScaler` to numerical features.
* Implemented preprocessing within the machine learning pipeline to maintain consistent transformations and prevent data leakage.

## Exploratory Data Analysis

The target variable was moderately imbalanced, with approximately **73.5% non-churn customers and 26.5% churn customers**. Therefore, model evaluation considered metrics beyond accuracy.

Key patterns identified during EDA:

* **Tenure:** Churned customers had substantially shorter tenure than non-churned customers.
* **Contract:** Month-to-month customers had considerably higher observed churn than customers on one-year or two-year contracts.
* **Internet Service:** Fiber optic customers showed the highest observed churn rate among the internet service groups.
* **Payment Method:** Electronic check customers showed the highest observed churn rate among payment methods.
* **Monthly Charges:** Churned customers had higher average monthly charges.
* **Correlation:** `tenure` and `TotalCharges` were strongly correlated (approximately 0.83).

These findings describe associations in the historical dataset and should not be interpreted as causal relationships.


## Model Development

The problem was formulated as a binary classification task.

The dataset was divided into training and testing sets using an **80/20 split** with stratification to preserve the churn distribution.

### Models Compared

Three classification algorithms were evaluated:

* Logistic Regression
* Random Forest
* Gradient Boosting

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

ROC-AUC was also used during cross-validation for model comparison.

### Cross-Validation Results

| Model               | Mean CV ROC-AUC |
| ------------------- | --------------: |
| Logistic Regression |           0.846 |
| Random Forest       |           0.825 |
| Gradient Boosting   |           0.847 |

Logistic Regression and Gradient Boosting produced similar cross-validation performance, while Random Forest performed lower on ROC-AUC.

Gradient Boosting was selected as the final model based on its slightly stronger cross-validation ROC-AUC and training-only threshold analysis.

### Final Model

The final model uses:

**Gradient Boosting Classifier**

with:

* `n_estimators = 200`
* `learning_rate = 0.05`
* `max_depth = 3`

The preprocessing and model were combined into a single Scikit-learn `Pipeline`.

### Classification Threshold

Instead of automatically using the default 0.50 threshold, the classification threshold was selected using **out-of-fold predictions from the training data**.

A threshold of **0.36** was selected based on the precision-recall trade-off and F1-score.

This threshold was determined without using the final test set, keeping the test set available for unbiased final evaluation.

## Final Test Performance

The final model was evaluated once on the untouched test set.

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **77.93%** |
| Precision | **56.80%** |
| Recall    | **70.32%** |
| F1-score  | **62.84%** |
| ROC-AUC   | **84.41%** |

The model correctly identified approximately **70% of the actual churners** in the test set.

The final confusion matrix was:

```text
[[835, 200],
 [111, 263]]
```

Where:

* **TN = 835**
* **FP = 200**
* **FN = 111**
* **TP = 263**

The model therefore identified 263 actual churners while missing 111 churners in the test set.


## Streamlit Application

A Streamlit web application was developed to allow users to enter customer information and receive a churn prediction.

### Application Workflow

```text
Customer Information
        ↓
Service Information
        ↓
Billing Information
        ↓
Trained ML Pipeline
        ↓
Churn Probability
        ↓
Prediction + Risk Level
```

The application:

* Accepts the same 19 input features used during model training.
* Uses the saved preprocessing and Gradient Boosting pipeline.
* Generates a churn probability for the entered customer profile.
* Applies the selected **0.36 classification threshold** to generate the final prediction.
* Displays a simple risk interpretation for business users.

The trained model is stored as:

```text
models/telco_churn_model.pkl
```

Prediction logic is also available in:

```text
src/predict.py
```


Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
Streamlit
Jupyter Notebook



## Business Insights

The analysis identified several customer characteristics associated with higher observed churn:

* Customers with shorter tenure showed higher churn rates.
* Month-to-month contracts had substantially higher observed churn than longer-term contracts.
* Fiber optic customers showed higher observed churn than other internet service groups.
* Electronic check customers had the highest observed churn rate among payment methods.
* Churned customers had higher average monthly charges.

These patterns can help a telecom business identify customer segments that may require closer monitoring or retention analysis.

The model can be used as a **decision-support tool** to prioritize customers for further investigation or retention activities. It should not be treated as a guarantee that an individual customer will churn.

## Limitations

* The dataset represents historical customer behavior and may not reflect current customer behavior.
* Observed relationships do not establish causation.
* The model's predictions depend on the quality and relevance of the available customer features.
* False positives and false negatives have different business costs, which were not explicitly provided for this project.
* The classification threshold of 0.36 was selected based on the precision-recall trade-off rather than a business-provided cost matrix.

## Conclusion

This project demonstrates a complete end-to-end machine learning workflow for customer churn prediction.

The workflow includes:

**Data Understanding → Data Cleaning → EDA → Feature Preparation → Model Development → Cross-Validation → Threshold Selection → Final Evaluation → Model Saving → Streamlit Deployment**

The final Gradient Boosting model achieved an ROC-AUC of **84.41%** on the held-out test set and identified approximately **70% of actual churners** at the selected classification threshold.

The resulting application provides a practical interface for generating customer-level churn predictions and demonstrates how a machine learning model can be integrated into a simple business-facing application.
