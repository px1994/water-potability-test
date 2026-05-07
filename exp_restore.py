import mlflow
import dagshub
from mlflow.tracking import MlflowClient

# Set DagsHub tracking URI
dagshub.init(repo_owner='pritesh13590', 
             repo_name='water-potability', 
             mlflow=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/pritesh13590/water-potability.mlflow")

# Create client
client = MlflowClient()

# Get all experiments including deleted ones
experiments = client.search_experiments(
    view_type=2
)

# Print experiments
for exp in experiments:

    print(
        "Name:", exp.name,
        "| ID:", exp.experiment_id,
        "| Stage:", exp.lifecycle_stage
    )  

client.restore_experiment("3")

print("Experiment restored successfully")