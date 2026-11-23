import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, task, get_run_logger

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    predict_model,
    save_model,
    load_model,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
)

# Fichiers concernés par les vérifications
FILES = [
    "model_pipeline.py",
    "main.py",
    "pipeline_prefect.py",
]


# Exécuter une commande avec le Python de l'environnement virtuel
def _run(module, *args):
    subprocess.run(
        [sys.executable, "-m", module, *args],
        check=True,
    )  # nosec B603


# =====================================================
# TASKS : gestion des dépendances et du modèle
# =====================================================


@task(name="install-dependencies")
def install_dependencies():
    logger = get_run_logger()
    _run("pip", "install", "-r", "requirements.txt")
    logger.info("Dependencies installed.")


@task(name="prepare-data")
def task_prepare_data():
    logger = get_run_logger()

    X_train, X_test, y_train, y_test = prepare_data(DEFAULT_DATA_PATH)

    logger.info("Data preparation complete.")

    return X_train, X_test, y_train, y_test


@task(name="train-model")
def task_train_model(X_train, y_train):
    logger = get_run_logger()

    model = train_model(X_train, y_train)

    logger.info("Training complete.")

    return model


@task(name="save-model")
def task_save_model(model):
    logger = get_run_logger()

    save_model(model, DEFAULT_MODEL_PATH)

    logger.info("Model saved.")


@task(name="load-model")
def task_load_model():
    logger = get_run_logger()

    model = load_model(DEFAULT_MODEL_PATH)

    logger.info("Model loaded.")

    return model


@task(name="evaluate-model")
def task_evaluate_model(model, X_test, y_test):
    logger = get_run_logger()

    evaluate_model(model, X_test, y_test)

    logger.info("Evaluation complete.")


@task(name="predict-model")
def task_predict_model(model, X_test):
    logger = get_run_logger()

    predictions, probabilities = predict_model(model, X_test)

    logger.info("Prediction complete.")
    logger.info(f"Nombre de prédictions : {len(predictions)}")

    return predictions, probabilities


# =====================================================
# TASKS : qualité et sécurité du code
# =====================================================


@task(name="format-code")
def format_code():
    logger = get_run_logger()

    _run("black", *FILES)

    logger.info("Code formatted.")


@task(name="check-quality")
def check_quality():
    logger = get_run_logger()

    _run(
        "flake8",
        "--max-line-length=88",
        *FILES,
    )

    logger.info("Code quality OK.")


@task(name="check-security")
def check_security():
    logger = get_run_logger()

    _run("bandit", *FILES)

    logger.info("Security check OK.")


@task(name="run-tests")
def run_tests():
    logger = get_run_logger()

    _run("pytest", "-v")

    logger.info("Unit tests passed.")


# =====================================================
# FLOWS : suivi de la qualité du code
# =====================================================


@flow(name="code", log_prints=True)
def flow_code():
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_tests()


# =====================================================
# FLOWS : Machine Learning
# =====================================================


@flow(name="train", log_prints=True)
def flow_train():
    X_train, X_test, y_train, y_test = task_prepare_data()

    model = task_train_model(X_train, y_train)

    task_save_model(model)


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    _, X_test, _, y_test = task_prepare_data()

    model = task_load_model()

    task_evaluate_model(model, X_test, y_test)

    # Prédiction
    task_predict_model(model, X_test)


@flow(name="all", log_prints=True)
def flow_all():
    # Vérification du code avant le pipeline ML
    flow_code()

    # Préparation des données
    X_train, X_test, y_train, y_test = task_prepare_data()

    # Entraînement
    model = task_train_model(X_train, y_train)

    # Sauvegarde
    task_save_model(model)

    # Évaluation
    task_evaluate_model(model, X_test, y_test)

    # Prédiction
    task_predict_model(model, X_test)


# =====================================================
# SCRIPT : choix du flow avec argparse
# =====================================================

FLOWS = {
    "all": flow_all,
    "train": flow_train,
    "evaluate": flow_evaluate,
    "code": flow_code,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline MLOps avec Prefect")

    parser.add_argument(
        "--flow",
        choices=FLOWS.keys(),
        required=True,
        help="Flow à exécuter",
    )

    args = parser.parse_args()

    FLOWS[args.flow]()
