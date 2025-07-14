import os
import gc
import yaml
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import mlflow
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from contextlib import nullcontext
from sklearn.model_selection import StratifiedKFold
import logging

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
# mlflow.set_tracking_uri("mlruns")
# mlflow_tracking_uri = 'http://127.0.0.1:5555'
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)


# Load data
df_train = pd.read_parquet("data/fd_train_df.parquet")
df_val = pd.read_parquet("data/fd_val_df.parquet")

X_train = df_train.drop(columns=["reaction_outcome"])
y_train = df_train["reaction_outcome"].astype(int)

X_val = df_val.drop(columns=["reaction_outcome"])
y_val = df_val["reaction_outcome"].astype(int)
del df_train
del df_val
gc.collect()


# Define models
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "ExtraTree": ExtraTreeClassifier(),
    # 'Bagging': BaggingClassifier(),
    # 'XGB': XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, verbosity=0),
    # 'LGBM': LGBMClassifier(),
}

# models = {
#     'DecisionTree': DecisionTreeClassifier(),
#     'XGB': XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, verbosity=0),
#     'LGBM': LGBMClassifier(),
# }

# Define hyperparameter spaces
model_grids = {
    "DecisionTree": {"max_depth": Integer(3, 20), "min_samples_split": Integer(2, 20)},
    "ExtraTree": {"max_depth": Integer(3, 20), "min_samples_split": Integer(2, 20)},
    # 'Bagging': {
    #     'n_estimators': Integer(10, 100),
    #     'max_samples': Real(0.5, 1.0)
    # },
    # 'XGB': {
    #     'n_estimators': Integer(50, 200),
    #     'max_depth': Integer(3, 10),
    #     'learning_rate': Real(0.01, 0.3, prior='log-uniform')
    # },
    # 'LGBM': {
    #     'n_estimators': Integer(50, 200),
    #     'max_depth': Integer(3, 10),
    #     'learning_rate': Real(0.01, 0.3, prior='log-uniform')
    # }
}

# Evaluate and tune
results = {}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

with (
    mlflow.start_run(run_name="model_comparison")
    if mlflow_tracking_uri
    else nullcontext()
):
    for name, model in models.items():
        logger.info(f"Tuning and evaluating {name}...")
        search = BayesSearchCV(
            estimator=model,
            search_spaces=model_grids[name],
            n_iter=25,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
            random_state=42,
        )

        with (
            mlflow.start_run(run_name=name, nested=True)
            if mlflow_tracking_uri
            else nullcontext()
        ):
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            # Evaluate on validation set
            preds = best_model.predict(X_val)
            f1 = f1_score(y_val, preds, average="weighted")
            precision = precision_score(y_val, preds, average="weighted")
            recall = recall_score(y_val, preds, average="weighted")

            results[name] = {
                "model": best_model,
                "params": search.best_params_,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }

            if mlflow_tracking_uri:
                mlflow.log_params(search.best_params_)
                mlflow.log_metrics({"f1": f1, "precision": precision, "recall": recall})
                mlflow.sklearn.log_model(best_model, artifact_path=name.lower())

            print(
                f"{name} F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
            )


# Save best model config

best_model_name = max(results, key=lambda x: results[x]["f1"])
best_model = results[best_model_name]["model"]
best_params = best_model.get_params()
best_f1 = float(results[best_model_name]["f1"])
best_precision = float(results[best_model_name]["precision"])
best_recall = float(results[best_model_name]["recall"])

print(f"\n Best Model: {best_model_name}")
print(f"   F1 Score: {best_f1:.4f}")
print(f"   Precision: {best_precision:.4f}")
print(f"   Recall: {best_recall:.4f}")

selected_features_dict = {"features": list(X_train.columns)}

model_config = {
    "model": {
        "name": "reaction_outcome_classifier",
        "best_model": best_model_name,
        "parameters": best_params,
        "f1_score": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "target_variable": "reaction_outcome",
        "feature_sets": selected_features_dict,
    }
}

config_path = os.getenv("MODEL_CONFIG")
os.makedirs(os.path.dirname(config_path), exist_ok=True)
with open(config_path, "w") as f:
    yaml.dump(model_config, f)

print(f"\n Saved model config to {config_path}")
