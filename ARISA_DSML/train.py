"""Functions to train model."""
from pathlib import Path

from catboost import CatBoostClassifier, Pool, cv
import joblib
import json
from loguru import logger
import optuna
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.metrics import f1_score, roc_auc_score

from ARISA_DSML.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    categorical,
    target,
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")


def run_hyperopt(X_train:pd.DataFrame, y_train:pd.DataFrame, categorical_indices:list[int], test_size:float=0.25, n_trials:int=20, overwrite:bool=False)->str|Path: # noqa: PLR0913
    """Run optuna hyperparameter tuning."""
    best_params_path = MODELS_DIR / "best_params.pkl"
    if not best_params_path.is_file() or overwrite:
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
        best_model = None  # Store best model
        best_params = None  # Store best parameters

        with mlflow.start_run(nested=True):
            def objective(trial:optuna.trial.Trial)->float:
                params = {
                    "depth": trial.suggest_int("depth", 2, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3),
                    "iterations": trial.suggest_int("iterations", 50, 300),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 100.0, log=True),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.01, 1),
                    "random_strength": trial.suggest_float("random_strength", 1e-5, 100.0, log=True),
                    "ignored_features": [0],
                }
                model = CatBoostClassifier(**params, verbose=0)
                model.fit(
                    X_train_opt,
                    y_train_opt,
                    eval_set=(X_val_opt, y_val_opt),
                    cat_features=categorical_indices,
                    early_stopping_rounds=50,
                )
                nonlocal best_model, best_params
                if study.best_trial == trial:
                    best_model = model
                    best_params = params
                return model.get_best_score()["validation"]["Logloss"]

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)

            if best_model is not None:
                mlflow.log_params(best_params)
                preds = best_model.predict(X_val_opt)
                probs = best_model.predict_proba(X_val_opt)
                f1 = f1_score(y_val_opt, preds)
                roc_auc = roc_auc_score(y_val_opt, probs[:, 1])
                mlflow.log_metrics({
                    "f1": f1,
                    "roc_auc": roc_auc,
                    "positive_class_mean_prob": probs[:, 1].mean(),
                })
                mlflow.log_artifact(best_params_path)

        joblib.dump(study.best_params, best_params_path)
        params = study.best_params
    else:
        params = joblib.load(best_params_path)
    logger.info("Best Parameters: " + json.dumps(params))

    return best_params_path


def get_or_create_experiment(experiment_name:str):
    """Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.



    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.



    Parameters
    ----------
    - experiment_name (str): Name of the MLflow experiment.



    Returns
    -------
    - str: ID of the existing or newly created MLflow experiment.



    """
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name)


def train_cv(X_train:pd.DataFrame, y_train:pd.DataFrame, categorical_indices:list[int], params:dict, eval_metric:str="F1", n:int=5)->str|Path:  # noqa: PLR0913
    """Do cross-validated training."""
    params["eval_metric"] = eval_metric
    params["loss_function"] = "Logloss"
    params["ignored_features"] = [0]  # ignore passengerid

    data = Pool(X_train, y_train, cat_features=categorical_indices)

    cv_results = cv(
        params=params,
        pool=data,
        fold_count=n,
        partition_random_seed=42,
        shuffle=True,
        plot=True,

    )

    cv_output_path = MODELS_DIR / "cv_results.csv"
    cv_results.to_csv(cv_output_path, index=False)

    return cv_output_path


def train(X_train:pd.DataFrame, y_train:pd.DataFrame, categorical_indices:list[int], params:dict|None, artifact_name:str="catboost_model_titanic", cv_metric_mean:float=0.0)->tuple[str|Path]:
    """Train model on full dataset."""
    if params is None:
        logger.info("Training model without tuned hyperparameters")
        params = {}

    with mlflow.start_run():
        params["ignored_features"] = [0]

        model = CatBoostClassifier(
            **params,
            verbose=True,
        )

        model.fit(
            X_train,
            y_train,
            verbose_eval=50,
            early_stopping_rounds=50,
            cat_features=categorical_indices,
            use_best_model=False,
            plot=True,
        )

        mlflow.log_params(params)
        params["feature_columns"] = X_train.columns
        model_path = MODELS_DIR / f"{artifact_name}.cbm"
        model.save_model(model_path)
        mlflow.log_artifact(model_path)
        mlflow.log_metric("f1_cv_mean", cv_metric_mean)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_params_path = MODELS_DIR / "model_params.pkl"
        joblib.dump(params, model_params_path)

    return (model_path, model_params_path)


def plot_error_scatter(  # noqa: PLR0913
        df_plot:pd.DataFrame,
        x:str="iterations",
        y:str="test-F1-mean",
        err:str="test-F1-std",
        name:str="",
        title:str="",
        xtitle:str="",
        ytitle:str="",
        yaxis_range:list[float]|None=None,
    )->None:
    """Plot plotly scatter plots with error areas."""
    # Create figure
    fig = go.Figure()

    if not len(name):
        name = y

    # Add mean performance line
    fig.add_trace(
        go.Scatter(
            x=df_plot[x], y=df_plot[y], mode="lines", name=name, line={"color": "blue"},
        ),
    )

    # Add shaded error region
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df_plot[y], df_plot[x][::-1]]),
            y=pd.concat([df_plot[y]+df_plot[err],
                         df_plot[y]-df_plot[err]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line={"color":"rgba(255, 255, 255, 0)"},
            showlegend=False,
        ),
    )

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        template="plotly_white",
    )

    if yaxis_range is not None:
        fig.update_layout(
            yaxis={"range": yaxis_range},
        )

    fig.show()
    fig.write_image(FIGURES_DIR / f"{y}_vs_{x}.png")


if __name__=="__main__":
    df_train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

    y_train = df_train.pop(target)
    X_train = df_train

    categorical_indices = [X_train.columns.get_loc(col) for col in categorical if col in X_train.columns]

    best_params_path = run_hyperopt(X_train, y_train, categorical_indices)
    params = joblib.load(best_params_path)
    cv_output_path = train_cv(X_train, y_train, categorical_indices, params)
    cv_results = pd.read_csv(cv_output_path)

    plot_error_scatter(
        df_plot=cv_results,
        name="Mean F1 Score",
        title="Cross-Validation (N=5) Mean F1 score with Error Bands",
        xtitle="Training Steps",
        ytitle="Performance Score",
        yaxis_range=[0.5, 1.0],
    )

    plot_error_scatter(
        cv_results,
        x="iterations",
        y="test-Logloss-mean",
        err="test-Logloss-std",
        name="Mean logloss",
        title="Cross-Validation (N=5) Mean Logloss with Error Bands",
        xtitle="Training Steps",
        ytitle="Logloss",
    )

    # here we would evaluate if model train run mean metric test score is above previous test score
    model_path, model_params_path = train(X_train, y_train, categorical_indices, params)

    cv_results = pd.read_csv(cv_output_path)
