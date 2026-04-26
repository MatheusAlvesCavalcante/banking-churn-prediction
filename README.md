# Banking Customer Churn: Machine Learning Pipeline 

This project implements a complete Data Science workflow to predict customer turnover (churn) in a banking institution. The goal is to provide a predictive tool that allows the bank to identify high-risk customers and apply targeted retention strategies.

##  Data Source
The dataset used in this project was sourced from **Kaggle**: 
[Bank Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)

##  Methodology and Project Structure

The project was developed following data science best practices, structured as follows:

### 1. Data Cleaning & Preprocessing
* **Feature Selection:** Removal of unique identifiers (`customer_id`, `surname`) that hold no predictive power.
* **Scaling:** Application of `StandardScaler` to normalize numerical variables with different ranges (e.g., `balance` and `estimated_salary`).
* **Categorical Encoding:** Implementation of `TargetEncoder` and binary mapping to handle features like geography and gender without increasing dimensionality excessively.

### 2. Feature Engineering
* **Pipeline Integration:** Utilized `ColumnTransformer` to create a robust preprocessing pipeline, ensuring that transformations are consistent across training and testing sets, preventing data leakage.

### 3. Model Benchmarking
Seven different algorithms were tested and validated using **K-Fold Cross-Validation** to identify the most robust learner:
* Logistic Regression (Baseline)
* SVM (Linear & RBF)
* K-Nearest Neighbors (KNN)
* Decision Tree
* **Random Forest** (Top Performer)
* XGBoost

##  Final Model Performance

The **Random Forest** model was selected after extensive hyperparameter tuning via `GridSearchCV`, showing the best results in terms of stability and predictive power:

* **ROC AUC:** 0.8636 (Cross-Validation Mean)
* **Accuracy:** ~86%
* **Strategic Focus:** High priority on **Recall** optimization to ensure maximum identification of customers likely to leave, supporting proactive business intervention.
