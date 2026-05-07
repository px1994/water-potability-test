# =========================================================
# IMPORT REQUIRED LIBRARIES
# =========================================================

# Data Handling
import pandas as pd
import numpy as np

# Model Saving
import os
import pickle

# MLflow + DagsHub
import mlflow
import mlflow.sklearn
import dagshub

# Scikit-learn
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# MLflow Signature
from mlflow.models import infer_signature

# =========================================================
# INITIALIZE DAGSHUB + MLFLOW
# =========================================================

dagshub.init(repo_owner='pritesh13590', 
             repo_name='water-potability', 
             mlflow=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/pritesh13590/water-potability.mlflow")

# Set Experiment Name
mlflow.set_experiment("Experiment 4")

# =========================================================
# LOAD DATASET
# =========================================================

data_path = r"c:/Data/project_datasets/water_potability.csv"

data = pd.read_csv(data_path)

print("Dataset Shape:", data.shape)

print(data.head())

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42
)

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

def fill_missing_with_mean(df):
    """
    Fill missing values using mean value
    """

    df = df.copy()

    for column in df.columns:

        if df[column].isnull().sum() > 0:

            mean_value = df[column].mean()

            df[column] = df[column].fillna(mean_value)

    return df


# Apply preprocessing
train_processed_data = fill_missing_with_mean(
    train_data
)

test_processed_data = fill_missing_with_mean(
    test_data
)

# =========================================================
# FEATURE / TARGET SPLIT
# =========================================================

X_train = train_processed_data.drop(
    columns=["Potability"],
    axis=1
)

y_train = train_processed_data["Potability"]

X_test = test_processed_data.drop(
    columns=["Potability"],
    axis=1
)

y_test = test_processed_data["Potability"]

# =========================================================
# DEFINE RANDOM FOREST MODEL
# =========================================================

rf = RandomForestClassifier(
    random_state=42
)

# =========================================================
# DEFINE HYPERPARAMETER SEARCH SPACE
# =========================================================

param_dist = {

    "n_estimators": [
        100,
        200,
        300,
        500,
        1000
    ],

    "max_depth": [
        None,
        4,
        5,
        6,
        10
    ]
}

# =========================================================
# RANDOMIZED SEARCH CV
# =========================================================

random_search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_dist,

    n_iter=10,

    cv=5,

    n_jobs=-1,

    verbose=2,

    random_state=42
)

# =========================================================
# START MLFLOW RUN
# =========================================================

with mlflow.start_run(
    run_name="Random_Forest_Tuning"
):

    # -----------------------------------------------------
    # TRAIN RANDOMIZED SEARCH
    # -----------------------------------------------------

    random_search.fit(X_train, y_train)

    # =====================================================
    # LOG ALL PARAMETER COMBINATIONS
    # =====================================================

    for i in range(
        len(random_search.cv_results_["params"])
    ):

        with mlflow.start_run(
            run_name=f"Combination_{i+1}",
            nested=True
        ):

            # Log parameters
            mlflow.log_params(
                random_search.cv_results_["params"][i]
            )

            # Log score
            mlflow.log_metric(
                "mean_test_score",
                random_search.cv_results_[
                    "mean_test_score"
                ][i]
            )

    # =====================================================
    # BEST PARAMETERS
    # =====================================================

    print(
        "\nBest Parameters:",
        random_search.best_params_
    )

    # Log best parameters
    mlflow.log_params(
        random_search.best_params_
    )

    # =====================================================
    # BEST MODEL
    # =====================================================

    best_rf = random_search.best_estimator_

    # Train best model
    best_rf.fit(X_train, y_train)

    # =====================================================
    # SAVE MODEL
    # =====================================================

    with open("models/model_exp4.pkl", "wb") as file:
        pickle.dump(best_rf, file)

    # =====================================================
    # LOAD MODEL
    # =====================================================

    with open("models/model_exp4.pkl", "rb") as file:
        model = pickle.load(file)

    # =====================================================
    # MAKE PREDICTIONS
    # =====================================================

    y_pred = model.predict(X_test)

    # =====================================================
    # EVALUATION METRICS
    # =====================================================

    acc = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print("\n===== MODEL PERFORMANCE =====")

    print("Accuracy :", acc)

    print("Precision:", precision)

    print("Recall   :", recall)

    print("F1-Score :", f1)

    # =====================================================
    # LOG METRICS
    # =====================================================

    mlflow.log_metric("accuracy", acc)

    mlflow.log_metric("precision", precision)

    mlflow.log_metric("recall", recall)

    mlflow.log_metric("f1_score", f1)

    # =====================================================
    # LOG DATASETS
    # =====================================================

    train_df = mlflow.data.from_pandas(
        train_processed_data
    )

    test_df = mlflow.data.from_pandas(
        test_processed_data
    )

    mlflow.log_input(
        train_df,
        context="train"
    )

    mlflow.log_input(
        test_df,
        context="test"
    )

    # =====================================================
    # INFER MODEL SIGNATURE
    # =====================================================

    signature = infer_signature(
        X_test,
        model.predict(X_test)
    )

    # =====================================================
    # LOG MODEL TO MLFLOW
    # =====================================================

    mlflow.sklearn.log_model(

        sk_model=model,

        artifact_path="Best_Model",

        signature=signature
    )

    # =====================================================
    # OPTIONAL: LOG SOURCE CODE FILE
    # =====================================================

    # mlflow.log_artifact(__file__)

    # =====================================================
    # SET TAGS
    # =====================================================

    mlflow.set_tag(
        "author",
        "pritesh"
    )

    mlflow.set_tag(
        "project",
        "Water Potability Prediction"
    )

# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n Random Forest tuning and MLflow logging completed successfully.")