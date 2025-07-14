import os
import argparse
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import yaml
from dotenv import load_dotenv
import logging
import platform
import sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

load_dotenv()

# ----------------------------- Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------- Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and register final classification model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=os.getenv("MODEL_CONFIG"),
        help="Path to model_config.yaml",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        default=os.getenv("DATASET_DIR"),
        help="Path to training Parquet file",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=False,
        default=os.getenv("MODEL_DIR"),
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI",
    )
    return parser.parse_args()


# ----------------------------- Model Factory
def get_model_instance(name, params):
    model_map = {
        "DecisionTree": DecisionTreeClassifier,
        "ExtraTree": ExtraTreesClassifier,
        # 'Bagging': BaggingClassifier,
        # 'XGB': XGBClassifier,
        # 'LGBM': LGBMClassifier
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)


# ----------------------------- Main Entry
def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]
    target = model_cfg["target_variable"]

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(model_cfg["name"])

    # Load data
    df_train = pd.read_parquet(args.data + "fd_train_df.parquet")
    df_val = pd.read_parquet(args.data + "fd_val_df.parquet")
    df = pd.concat([df_train, df_val])
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Instantiate model
    model = get_model_instance(model_cfg["best_model"], model_cfg["parameters"])

    # Train and log
    with mlflow.start_run(run_name="final_classification_training"):
        logger.info(f"Training model: {model_cfg['best_model']}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        precision = float(precision_score(y_test, y_pred, average="weighted"))
        recall = float(recall_score(y_test, y_pred, average="weighted"))

        mlflow.log_params(model_cfg["parameters"])
        mlflow.log_metrics({"f1": f1, "precision": precision, "recall": recall})

        mlflow.sklearn.log_model(model, "final_model")
        model_name = model_cfg["name"]
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/final_model"

        logger.info("Registering model to MLflow Model Registry...")
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.RestException:
            pass  # already exists

        model_version = client.create_model_version(
            name=model_name, source=model_uri, run_id=mlflow.active_run().info.run_id
        )

        client.transition_model_version_stage(
            name=model_name, version=model_version.version, stage="Staging"
        )

        description = (
            f"Classification model for predicting {target}.\n"
            f"Algorithm: {model_cfg['best_model']}\n"
            f"Hyperparameters: {model_cfg['parameters']}\n"
            f"Features used: All columns except target\n"
            f"Target variable: {target}\n"
            f"Dataset: {args.data}\n"
            f"Performance:\n"
            f"  - F1: {f1:.4f}\n"
            f"  - Precision: {precision:.4f}\n"
            f"  - Recall: {recall:.4f}"
        )
        client.update_registered_model(name=model_name, description=description)

        client.set_registered_model_tag(
            model_name, "algorithm", model_cfg["best_model"]
        )
        client.set_registered_model_tag(
            model_name, "hyperparameters", str(model_cfg["parameters"])
        )
        client.set_registered_model_tag(model_name, "target_variable", target)
        client.set_registered_model_tag(model_name, "training_dataset", args.data)

        # Dependency tags
        deps = {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": XGBClassifier().__module__.split(".")[0],
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
        for k, v in deps.items():
            client.set_registered_model_tag(model_name, k, v)

        # Save model locally
        os.makedirs(f"{args.models_dir}/trained", exist_ok=True)
        save_path = f"{args.models_dir}/trained/{model_name}.pkl"
        joblib.dump(model, save_path)

        logger.info(f"Model saved to: {save_path}")
        logger.info(
            f"Final F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
