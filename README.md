# Banking Customer Churn Prediction 

**Leveraging Random Forest and Hyperparameter Optimization for Strategic Retention.**

This project implements an end-to-end Machine Learning pipeline to predict the probability of customer churn in a banking environment. The primary focus was on statistical robustness and business-driven metric optimization. Developed as part of my Data Science studies at the **Federal University of Ceará (UFC)**.

##  Key Performance Indicators
* **ROC AUC:** 0.8636
* **Accuracy:** 86%
* **Final Model:** Random Forest Classifier

##  Technical Implementation

* **Feature Engineering:** Developed financial ratios (e.g., Balance-to-Salary) to uncover deeper behavioral patterns.
* **Preprocessing Pipeline:** Automated workflow using `Target Encoding` for categorical data and `StandardScaler` for numerical consistency.
* **Validation Strategy:** Employed `Stratified K-Fold` (5 folds) to handle class imbalance and ensure model generalization.
* **Model Tuning:** Executed `GridSearchCV` to optimize parameters including `max_depth`, `min_samples_leaf`, and `class_weight`.

##  Business Insights
The final model was fine-tuned to balance Precision and Recall. By optimizing the decision threshold, the institution can identify at-risk customers with high confidence, directly supporting proactive marketing and retention efforts.

---
