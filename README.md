📊 Customer Churn Prediction and Retention Model
📌 Project Overview

This project develops a machine learning model to predict customer churn using customer interaction and call behavior data. The objective is to identify customers at high risk of churn and uncover behavioral patterns that drive customer attrition.

The analysis includes exploratory data analysis (EDA), feature engineering, multicollinearity diagnostics, and model comparison across multiple algorithms.


🎯 Problem Statement

Customer churn is a critical business challenge. The goal of this project is to:

- Predict whether a customer will churn

- Identify behavioral signals associated with churn

- Compare multiple machine learning models

- Provide actionable insights for retention strategies

Target Variable:

- Churn (Binary: 0 = Retained, 1 = Churned)


📂 Dataset Overview

The dataset includes:

- Customer tenure (duration)

- Call behavior metrics

- Talk time and hold time statistics

- Call types

- Reason descriptions for service interactions

- Engineered behavioral features

The data was preprocessed and encoded before modeling.


🔎 Exploratory Data Analysis (EDA)

Key areas explored:

- Distribution of churn vs non-churn customers

- Call frequency by churn status

- Talk time and hold time patterns

- Relationship between service friction and churn

EDA revealed strong signals linking higher service friction (long hold times) with increased churn probability.


🛠 Feature Engineering

To improve predictive performance, several behavioral features were engineered:

- Total Talk Time

- Total Hold Time

- Hold Ratio (avg_hold_time / avg_talk_time)

- Hold Severity

- Frustration Score

- Engagement Score

- Aggregated customer-level statistics

- Encoded categorical variables

Multicollinearity was evaluated using Variance Inflation Factor (VIF) to ensure model stability for linear models.


🤖 Modeling Approach
1️⃣ Data Split

Train/Test split using train_test_split

2️⃣ Models Trained

- Logistic Regression

- Random Forest

- XGBoost

3️⃣ Evaluation Metrics

Models were evaluated using:

- Accuracy

- Precision

- Recall

- F1 Score

- ROC-AUC

ROC-AUC was emphasized due to its robustness in binary classification problems.

4️⃣ Hyperparameter Tuning


📈 Model Comparison

Multiple models were compared to assess:

- Linear vs tree-based performance

- Impact of engineered features

- Sensitivity to multicollinearity

- Tree-based models (Random Forest and XGBoost) demonstrated stronger performance due to their ability to capture nonlinear relationships and feature interactions.


🔑 Key Insights

- Customers with higher hold times are more likely to churn.

- Asides those who churned for vague reasons, competitor deals is the highest reason people churn.

- Loyalty calls is the highest reason people call after the unknown category. 

- Behavioral interaction features improve predictive power.
  

💼 Business Implications

The model can be used to:

- Identify high-risk customers in advance

- Trigger proactive retention campaigns

- Reduce call center friction

- Optimize customer service performance

- Early intervention strategies targeting high-risk customers could significantly reduce churn rates.
