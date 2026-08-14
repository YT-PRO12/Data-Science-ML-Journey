# %% [markdown]
# # 🤖 Scikit-Learn: Machine Learning Workflow
#
# A practical introduction to the Scikit-Learn workflow:
#
# Data → Features & Target → Train/Test Split → Model → Fit → Predict
# → Evaluate → Cross-Validate → Tune → Save → Pipeline
#
# ## 🎯 Learning Objectives
# - Prepare data for Machine Learning
# - Separate features and target
# - Split data into training and testing sets
# - Train classification and regression models
# - Make predictions
# - Evaluate models with suitable metrics
# - Use cross-validation
# - Tune hyperparameters
# - Save and reload a trained model
# - Build a preprocessing + modelling pipeline
#
# ## 🛠️ Tools
# Python • NumPy • Pandas • Matplotlib • Scikit-Learn

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(42)

print("Libraries imported successfully!")

# %% [markdown]
# ## 1. Get the Data Ready
#
# Scikit-Learn provides built-in datasets that make this notebook
# self-contained and reproducible.
#
# We use:
# - Breast Cancer Wisconsin dataset → classification
# - Diabetes dataset → regression

# %%
# Classification dataset
cancer = load_breast_cancer()

X_classification = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)
y_classification = pd.Series(cancer.target, name="target")

print("Classification dataset shape:", X_classification.shape)
display(X_classification.head())

# %%
# Regression dataset
diabetes = load_diabetes()

X_regression = pd.DataFrame(
    diabetes.data,
    columns=diabetes.feature_names
)
y_regression = pd.Series(diabetes.target, name="target")

print("Regression dataset shape:", X_regression.shape)
display(X_regression.head())

# %% [markdown]
# ## 2. Features and Target
#
# **Features (X)** are the input variables used by the model.
#
# **Target (y)** is the value the model learns to predict.

# %%
print("Classification features:", X_classification.shape)
print("Classification target:", y_classification.shape)

print("\nRegression features:", X_regression.shape)
print("Regression target:", y_regression.shape)

# %% [markdown]
# ## 3. Train/Test Split
#
# Training data is used to learn patterns.
# Testing data is kept separate to evaluate generalization.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_classification,
    y_classification,
    test_size=0.20,
    random_state=42,
    stratify=y_classification
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# %% [markdown]
# ## 4. Choose an Estimator
#
# Scikit-Learn calls Machine Learning models **estimators**.
#
# For this classification example we use Random Forest.

# %%
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print(clf)

# %% [markdown]
# ## 5. Fit the Model

# %%
clf.fit(X_train, y_train)

print("Model training completed!")

# %% [markdown]
# ## 6. Make Predictions

# %%
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred
})

display(results.head(10))

# %% [markdown]
# ## 7. Evaluate the Classification Model

# %%
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Classification Performance")
print("---------------------------")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")

# %%
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=cancer.target_names
))

# %% [markdown]
# ## 8. Cross-Validation
#
# Cross-validation evaluates the model across multiple train/validation
# splits and gives a more reliable estimate than one split alone.

# %%
cv_scores = cross_val_score(
    clf,
    X_classification,
    y_classification,
    cv=5,
    scoring="accuracy"
)

print("Cross-validation scores:", np.round(cv_scores, 3))
print("Mean CV accuracy:", round(cv_scores.mean(), 3))

# %% [markdown]
# ## 9. Regression Workflow
#
# The same general Scikit-Learn workflow can be applied to regression.

# %%
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regression,
    y_regression,
    test_size=0.20,
    random_state=42
)

reg_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

reg_model.fit(X_train_reg, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg)

# %%
mae = mean_absolute_error(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print("Regression Performance")
print("----------------------")
print(f"MAE : {mae:.3f}")
print(f"MSE : {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²  : {r2:.3f}")

# %% [markdown]
# ## 10. Actual vs Predicted

# %%
plt.figure(figsize=(8, 5))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7)

plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Actual vs Predicted — Regression")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Hyperparameter Tuning
#
# Hyperparameters are settings chosen before training.
# RandomizedSearchCV tests different combinations using cross-validation.

# %%
param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

print("Best Parameters:")
print(search.best_params_)

print("\nBest Cross-Validation Score:")
print(round(search.best_score_, 3))

# %% [markdown]
# ## 12. Evaluate the Tuned Model

# %%
tuned_model = search.best_estimator_

tuned_predictions = tuned_model.predict(X_test)

print(
    "Tuned Model Accuracy:",
    round(accuracy_score(y_test, tuned_predictions), 3)
)

# %% [markdown]
# ## 13. Save and Reload the Model
#
# A trained model can be saved for later use.

# %%
import joblib

model_path = "ml_fundamentals_random_forest.joblib"

joblib.dump(tuned_model, model_path)

print(f"Model saved as: {model_path}")

# %%
loaded_model = joblib.load(model_path)

loaded_predictions = loaded_model.predict(X_test)

print(
    "Loaded Model Accuracy:",
    round(accuracy_score(y_test, loaded_predictions), 3)
)

# %% [markdown]
# ## 14. Build a Preprocessing + Modelling Pipeline
#
# Real-world datasets often contain:
# - Missing values
# - Numerical features
# - Categorical features
#
# A Pipeline keeps preprocessing and modelling together and helps
# prevent data leakage.

# %%
sample_data = pd.DataFrame({
    "Age": [22, 35, np.nan, 42, 29, 51, 31, np.nan],
    "Experience": [1, 8, 3, np.nan, 4, 20, 7, 2],
    "Department": [
        "IT", "HR", "IT", "Finance",
        "HR", "IT", "Finance", "IT"
    ],
    "Salary": [35, 60, 45, 70, 55, 95, 65, 50]
})

display(sample_data)

# %%
X = sample_data.drop("Salary", axis=1)
y = sample_data["Salary"]

numeric_features = ["Age", "Experience"]
categorical_features = ["Department"]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# %%
X_train_pipe, X_test_pipe, y_train_pipe, y_test_pipe = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model_pipeline.fit(X_train_pipe, y_train_pipe)

pipeline_predictions = model_pipeline.predict(X_test_pipe)

print("Pipeline predictions:", np.round(pipeline_predictions, 2))

# %% [markdown]
# ## 🔄 Complete Machine Learning Workflow
#
# ```text
# Data
#   ↓
# Exploration
#   ↓
# Features & Target
#   ↓
# Train/Test Split
#   ↓
# Choose Model
#   ↓
# Fit
#   ↓
# Predict
#   ↓
# Evaluate
#   ↓
# Cross-Validate
#   ↓
# Tune Hyperparameters
#   ↓
# Save Model
#   ↓
# Deploy / Use for New Data
# ```
#
# ## 🎯 Key Takeaways
#
# - `fit()` trains a model.
# - `predict()` generates predictions.
# - Classification and regression require different evaluation metrics.
# - Cross-validation provides a stronger estimate of model performance.
# - Hyperparameter tuning can improve a baseline model.
# - `Pipeline` combines preprocessing and modelling into one workflow.
# - Saving a model allows it to be reused without retraining.
#
# ## 🚀 Next Step
#
# **Data Preprocessing:** missing values, categorical encoding,
# feature scaling, and robust preprocessing pipelines.
