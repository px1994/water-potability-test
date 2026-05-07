# =========================================================
# IMPORT REQUIRED LIBRARIES
# =========================================================

# Data Handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model Saving
import pickle

# MLflow + DagsHub
import mlflow
import mlflow.sklearn
import dagshub

# Train Test Split
from sklearn.model_selection import train_test_split

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Evaluation Metrics
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

# Set MLflow experiment
mlflow.set_experiment("Experiment 3")

# =========================================================
# LOAD DATASET
# =========================================================

data_path = pd.read_csv("c:/Data/project_datasets/water_potability.csv")

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
    Fill missing values with mean value
    """

    df = df.copy()

    for column in df.columns:

        if df[column].isnull().sum() > 0:

            mean_value = df[column].mean()

            df[column] = df[column].fillna(mean_value)

    return df


# Apply preprocessing
train_processed_data = fill_missing_with_mean(train_data)

test_processed_data = fill_missing_with_mean(test_data)

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
# DEFINE MACHINE LEARNING MODELS
# =========================================================

models = {

    "Logistic_Regression": LogisticRegression(),

    "Random_Forest": RandomForestClassifier(
        random_state=42
    ),

    "Support_Vector_Classifier": SVC(),

    "Decision_Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNearest_Neighbors": KNeighborsClassifier(),

    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss"
    )
}

# =========================================================
# START PARENT MLFLOW RUN
# =========================================================

with mlflow.start_run(
    run_name="Water_Potability_Models_Experiment"
):

    # -----------------------------------------------------
    # LOOP THROUGH EACH MODEL
    # -----------------------------------------------------

    for model_name, model in models.items():

        print(f"\nTraining Model: {model_name}")

        # -------------------------------------------------
        # START CHILD RUN
        # -------------------------------------------------

        with mlflow.start_run(
            run_name=model_name,
            nested=True
        ):

            # =============================================
            # TRAIN MODEL
            # =============================================

            model.fit(X_train, y_train)

            # =============================================
            # SAVE MODEL
            # =============================================

            model_filename = f"{model_name}.pkl"

            with open(model_filename, "wb") as file:
                pickle.dump(model, file)

            # =============================================
            # MAKE PREDICTIONS
            # =============================================

            y_pred = model.predict(X_test)

            # =============================================
            # CALCULATE METRICS
            # =============================================

            acc = accuracy_score(y_test, y_pred)

            precision = precision_score(y_test, y_pred)

            recall = recall_score(y_test, y_pred)

            f1 = f1_score(y_test, y_pred)

            # =============================================
            # PRINT RESULTS
            # =============================================

            print("Accuracy :", acc)
            print("Precision:", precision)
            print("Recall   :", recall)
            print("F1-Score :", f1)

            # =============================================
            # LOG METRICS TO MLFLOW
            # =============================================

            mlflow.log_metric("accuracy", acc)

            mlflow.log_metric("precision", precision)

            mlflow.log_metric("recall", recall)

            mlflow.log_metric("f1_score", f1)

            # =============================================
            # LOG PARAMETERS
            # =============================================

            mlflow.log_param("model_name", model_name)

            # =============================================
            # GENERATE CONFUSION MATRIX
            # =============================================

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

            plt.title(
                f"Confusion Matrix - {model_name}"
            )

            # =============================================
            # SAVE CONFUSION MATRIX IMAGE
            # =============================================

            cm_filename = (
                f"confusion_matrix_{model_name}.png"
            )

            plt.savefig(cm_filename)

            plt.close()

            # =============================================
            # LOG ARTIFACTS
            # =============================================

            mlflow.log_artifact(cm_filename)

            # =============================================
            # LOG MODEL
            # =============================================

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name
            )

    # =====================================================
    # OPTIONAL: LOG SOURCE CODE FILE
    # =====================================================

    # mlflow.log_artifact(__file__)

    # =====================================================
    # SET TAGS
    # =====================================================

    mlflow.set_tag("author", "datathinkers")

    mlflow.set_tag(
        "project",
        "Water Potability Prediction"
    )

# =========================================================
# FINAL MESSAGE
# =========================================================

print(
    "\nAll models have been trained and logged successfully."
)