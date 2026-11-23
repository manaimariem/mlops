import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Machine Learning pour le diabète"
    )

    parser.add_argument(
        "--step",
        choices=["prepare", "train", "evaluate", "all"],
        required=True,
        help="Étape à exécuter",
    )

    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Chemin du fichier CSV",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Chemin du modèle",
    )

    args = parser.parse_args()

    print(">>> Préparation des données...")
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    print(f"X_train : {X_train.shape} | X_test : {X_test.shape}")

    print(
        f"% réadmis - train : {y_train.mean() * 100:.2f} % "
        f"| test : {y_test.mean() * 100:.2f} %"
    )

    if args.step == "prepare":
        return

    if args.step in ("train", "all"):
        print(">>> Entraînement du modèle...")
        model = train_model(X_train, y_train)
        save_model(model, args.model)

    else:
        model = load_model(args.model)

    if args.step in ("evaluate", "all"):
        print(">>> Évaluation...")
        evaluate_model(model, X_test, y_test)

    if args.step == "train":
        print(
            "Modèle entraîné et sauvegardé. " "Lancez --step evaluate pour l'évaluer."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erreur : {exc}")
        raise
