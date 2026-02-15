import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
    StratifiedKFold,
    GridSearchCV,
)
from scipy.stats import uniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    auc,
)
import os

# Load datasets
df = pd.read_csv("/Users/macintosh/Desktop/VS code/combined_data.csv")

# Fill NaNs for max hold time with 0
df["max_hold_time"] = df["max_hold_time"].fillna(0)
df["std_hold_time"] = df["std_hold_time"].fillna(0)
df["talk_time_seconds_std"] = df["talk_time_seconds_std"].fillna(0)

# Encode categorical variables for correlation analysis
df_encoded = df.copy()
df_encoded["call_type_encoded"] = df_encoded["call_type"].astype("category").cat.codes
df_encoded["reason_description_insight_encoded"] = (
    df_encoded["reason_description_insight"].astype("category").cat.codes
)

# Feature Engineering: Create new features based on domain knowledge
df_encoded["total_talk_time"] = (
    df_encoded["avg_talk_time_seconds"] * df_encoded["num_calls"]
)
df_encoded["total_hold_time"] = (
    df_encoded["avg_hold_time_seconds"] * df_encoded["num_calls"]
)
df_encoded["hold_ratio"] = df_encoded["avg_hold_time_seconds"] / (
    df_encoded["avg_talk_time_seconds"] + 1e-6
)
df_encoded["hold_severity"] = df_encoded["max_hold_time"] / (
    df_encoded["avg_hold_time_seconds"] + 1e-6
)
df_encoded["frustration_score"] = (
    df_encoded["avg_hold_time_seconds"] * df_encoded["num_calls"]
) / (df_encoded["avg_talk_time_seconds"] + 1e-6)
df_encoded["engagement_score"] = df_encoded["total_talk_time"] / (
    df_encoded["duration"] + 1e-6
)

# Split data into training and testing sets
X = df_encoded.drop(
    columns=[
        "Churn",
        "cease_completed_date",
        "cease_placed_date",
        "unique_customer_identifier",
        "call_type",
        "reason_description_insight",
        "reason_description",
        "duration",
        "num_calls",
        "total_talk_time",
        "total_hold_time",
        "max_hold_time",
        "engagement_score",
        "duration",
    ]
)
y = df_encoded["Churn"]

X_train, X_other, y_train, y_other = train_test_split(
    X, y, train_size=0.7, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X_other, y_other, test_size=0.3, random_state=42
)

# Normalize data
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
X_val = ss.transform(X_val)


# Evaluate function
def evaluate(X, y, model, subset=""):

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)

    print(f"\nEvaluation Metrics for {subset}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Churn): {prec:.4f}")
    print(f"Recall (Churn): {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {subset}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {subset}")
    plt.legend()
    plt.show()

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        importances = None

    if importances is not None:
        feature_names = (
            X.columns
            if isinstance(X, pd.DataFrame)
            else [f"Feature_{i}" for i in range(X.shape[1])]
        )

        feat_imp = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values(by="Importance", ascending=False)

        sns.barplot(data=feat_imp.head(10), x="Importance", y="Feature")
        plt.title(f"Top 10 Feature Importance - {subset}")
        plt.show()

    print(
        "\nClassification Report:\n",
        classification_report(y, y_pred, target_names=["Not Churned", "Churned"]),
    )

    return acc, prec, rec, f1, roc_auc

    # Cross-validation score
    def cross_validate(model, X, y, cv=5):
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f}")
        return scores.mean()


# Build Logistic Regression model
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train, y_train)

# Evaluate the model on training, validation, and test sets
evaluate(X_train, y_train, lr, subset="Training Set")
evaluate(X_val, y_val, lr, subset="Validation Set")

# Build Random Forest model
rf = RandomForestClassifier(
    n_estimators=200, max_depth=None, class_weight="balanced", random_state=42, n_jobs=1
)
rf.fit(X_train, y_train)

# Evaluate the model on training, validation, and test sets
evaluate(X_train, y_train, rf, subset="Training Set")
evaluate(X_val, y_val, rf, subset="Validation Set")

# Build XGBoost model
xgb = XGBClassifier(
    class_weight="balanced",
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
xgb.fit(X_train, y_train)

# Evaluate the model on training, validation, and test sets
evaluate(X_train, y_train, xgb, subset="Training Set")
evaluate(X_val, y_val, xgb, subset="Validation Set")

# Find best XGBoost model using randomized search CV
# Parameter distribution
param_dist_xgb = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10, 15],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5],
    "min_child_weight": [1, 3, 5, 7],
    "scale_pos_weight": [1, 3, 5, 10],
}

# Randomized Search
random_search_xgb = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist_xgb,
    n_iter=40,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2,
    random_state=42,
)

# Fit
random_search_xgb.fit(X_train, y_train)

# Best model
print("Best Parameters:", random_search_xgb.best_params_)
best_rs_model_xgb = random_search_xgb.best_estimator_

# Predictions
y_pred_xgb = best_rs_model_xgb.predict(X_test)
y_prob_xgb = best_rs_model_xgb.predict_proba(X_test)[:, 1]

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob_xgb))

# Find best Logistic Regression model using randomized search CV
# Parameter distribution
param_dist_lr = {
    "C": np.logspace(-4, 4, 20),
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
    "class_weight": [None, "balanced"],
}

# Randomized Search
random_search_lr = RandomizedSearchCV(
    estimator=lr,
    param_distributions=param_dist_lr,
    n_iter=30,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2,
    random_state=42,
)

# Fit
random_search_lr.fit(X_train, y_train)

# Best model
print("Best Parameters:", random_search_lr.best_params_)
best_rs_lr_model = random_search_lr.best_estimator_

# Predictions
y_pred_lr = best_rs_lr_model.predict(X_test)
y_prob_lr = best_rs_lr_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob_lr))

# Find best Random Forest model using randomized search CV
# Parameter distribution
param_dist_rf = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10, 20, 50, 100],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "max_features": ["sqrt", "log2", None],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced", "balanced_subsample"],
    "bootstrap": [True, False],
}

# Randomized Search
random_search_rf = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist_rf,
    n_iter=30,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2,
    random_state=42,
)

# Fit
random_search_rf.fit(X_train, y_train)

# Best model
print("Best Parameters:", random_search_rf.best_params_)
best_rs_rf_model = random_search_rf.best_estimator_

# Predictions
y_pred_rf = best_rs_rf_model.predict(X_test)
y_prob_rf = best_rs_rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob_rf))

# Compare AUC scores of all models
print(
    "Logistic AUC:", roc_auc_score(y_test, best_rs_lr_model.predict_proba(X_test)[:, 1])
)
print(
    "Random Forest AUC:",
    roc_auc_score(y_test, best_rs_rf_model.predict_proba(X_test)[:, 1]),
)
print(
    "XGBoost AUC:", roc_auc_score(y_test, best_rs_model_xgb.predict_proba(X_test)[:, 1])
)

# Run models again using best model
# Evaluate the random forest model on test set
evaluate(X_test, y_test, best_rs_rf_model, subset="Test Set")

# Evaluate the Logistic Regression model on test set
evaluate(X_test, y_test, best_rs_lr_model, subset="Test Set")

# Evaluate the XGBoost model on test set
evaluate(X_test, y_test, best_rs_model_xgb, subset="Test Set")
