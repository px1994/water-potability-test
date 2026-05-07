# =========================================================
# IMPORT REQUIRED LIBRARIES
# =========================================================

import pandas as pd
import numpy as np

#Model Saving
import pickle
import os

# MLflow + DagsHub
import mlflow
import mlflow.sklearn
import dagshub

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# INITIALIZE DAGSHUB + MLFLOW
# =========================================================

dagshub.init(repo_owner='pritesh13590', 
             repo_name='water-potability', 
             mlflow=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/pritesh13590/water-potability.mlflow")

# Create / Set experiment
mlflow.set_experiment("Experiment 1")

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

def fill_missing_with_median(df):
    """
    Fill missing values with median of each column
    """

    df = df.copy()

    for column in df.columns:

        if df[column].isnull().sum() > 0:

            median_value = df[column].median()

            df[column] = df[column].fillna(median_value)

    return df


# Apply preprocessing
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# =========================================================
# FEATURE / TARGET SPLIT
# =========================================================

X_train = train_processed_data.drop("Potability", axis=1)
y_train = train_processed_data["Potability"]

X_test = test_processed_data.drop("Potability", axis=1)
y_test = test_processed_data["Potability"]

# =========================================================
# MODEL PARAMETERS
# =========================================================

n_estimators = 100

# =========================================================
# START MLFLOW RUN
# =========================================================

with mlflow.start_run():

    # -----------------------------------------------------
    # MODEL TRAINING
    # -----------------------------------------------------

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

    clf.fit(X_train, y_train)

    # -----------------------------------------------------
    # SAVE MODEL USING PICKLE
    # -----------------------------------------------------

    # Create folder if not exists
    os.makedirs("models", exist_ok=True)

    with open("models/model_exp1.pkl", "wb") as file:
        pickle.dump(clf, file)

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    with open("models/model_exp1.pkl", "rb") as file:
        model = pickle.load(file)

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    y_pred = model.predict(X_test)

    # -----------------------------------------------------
    # EVALUATION METRICS
    # -----------------------------------------------------

    acc = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    # -----------------------------------------------------
    # PRINT METRICS
    # -----------------------------------------------------

    print("\n===== MODEL PERFORMANCE =====")

    print("Accuracy :", acc)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-Score :", f1)

    # -----------------------------------------------------
    # LOG METRICS TO MLFLOW
    # -----------------------------------------------------

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # -----------------------------------------------------
    # LOG PARAMETERS
    # -----------------------------------------------------

    mlflow.log_param("n_estimators", n_estimators)

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save image
    plt.savefig("confusion_matrix.png")

    # -----------------------------------------------------
    # LOG ARTIFACTS
    # -----------------------------------------------------

    mlflow.log_artifact("confusion_matrix.png")

    # Log model
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="RandomForestClassifier"
    )

    # Optional: log source code file
    # mlflow.log_artifact(__file__)

    # -----------------------------------------------------
    # SET TAGS
    # -----------------------------------------------------

    mlflow.set_tag("author", "pritesh")
    mlflow.set_tag("model_type", "RandomForest")

print("\n MLflow experiment completed successfully.")