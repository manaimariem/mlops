from model_pipeline import prepare_data, DEFAULT_DATA_PATH   # fonction et chemin du projet

# Vérifie que la préparation des données retourne des données non vides
def test_prepare_data_not_empty():
    X_train, X_test, y_train, y_test = prepare_data(DEFAULT_DATA_PATH)
    assert len(X_train) > 0
    assert len(X_test) > 0
