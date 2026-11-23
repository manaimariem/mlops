"""

model_pipeline.py
-----------------
Pipeline ML modulaire : prédiction de la réadmission à 30 jours
(dataset Diabetes 130-US hospitals, 1999-2008).

Fonctions principales :
    prepare_data()   : charger, nettoyer et séparer les données (train/test)
    train_model()    : entraîner le Pipeline (prétraitement + Random Forest)
    evaluate_model() : calculer les métriques sur le jeu de test
    save_model()     : sauvegarder le modèle avec joblib
    load_model()     : recharger un modèle sauvegardé
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Constantes (reprises du notebook)
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = "diabetic_data.csv"
DEFAULT_MODEL_PATH = "model.joblib"
RANDOM_STATE = 42

num_features = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "total_visits",
    "nb_med_changes",
    "nb_meds_diabete",
]
bin_features = ["deja_hospitalise"]
cat_features = [
    "race",
    "gender",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "insulin",
    "change",
    "diabetesMed",
    "admission_type",
    "admission_source",
    "discharge_disposition",
]

# Médicaments conservés (prescrits dans plus de 5 % des séjours)
MEDS_GARDER = [
    "insulin",
    "metformin",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "glimepiride",
]

# Codes "décédé" de discharge_disposition_id
CODES_DECES = [11, 19, 20, 21]


# ---------------------------------------------------------------------------
# Fonctions de regroupement (règles fixes, sans fuite de données)
# ---------------------------------------------------------------------------
def groupe_diag(code):
    """Regroupe un code diagnostic ICD-9 en grand chapitre médical."""
    if code == "Missing":
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    x = float(code)
    if int(x) == 250:
        return "Diabetes"
    if 390 <= x <= 459 or int(x) == 785:
        return "Circulatory"
    if 460 <= x <= 519 or int(x) == 786:
        return "Respiratory"
    if 520 <= x <= 579 or int(x) == 787:
        return "Digestive"
    if 580 <= x <= 629 or int(x) == 788:
        return "Genitourinary"
    if 710 <= x <= 739:
        return "Musculoskeletal"
    if 800 <= x <= 999:
        return "Injury"
    if 140 <= x <= 239:
        return "Neoplasms"
    return "Other"


def groupe_specialite(s):
    """Regroupe la spécialité médicale en quelques catégories."""
    s = str(s)
    if s in [
        "Unknown",
        "InternalMedicine",
        "Emergency/Trauma",
        "Family/GeneralPractice",
    ]:
        return s
    if s.startswith("Cardiology"):
        return "Cardiology"
    if "Surg" in s:
        return "Surgery"
    if s.startswith("Orthopedics"):
        return "Orthopedics"
    return "Other"


# ---------------------------------------------------------------------------
# 1. Préparation des données
# ---------------------------------------------------------------------------
def clean_data(df):
    """
    Applique les règles fixes de nettoyage et de feature engineering
    (parties 4.1 à 4.5 du notebook).

    Args:
        df (pd.DataFrame): données brutes lues depuis diabetic_data.csv.
    Returns:
        pd.DataFrame: données nettoyées, avec la cible binaire 'readmit_30'.
    """
    df = df.copy()

    # 4.1 Qualité : patients décédés, sexe invalide, colonnes constantes
    df = df[~df["discharge_disposition_id"].isin(CODES_DECES)]
    df = df[df["gender"] != "Unknown/Invalid"]
    df = df.drop(columns=["examide", "citoglipton"])

    # 4.2 Valeurs manquantes
    df = df.drop(columns=["weight", "payer_code"])
    df["race"] = df["race"].fillna("Unknown")
    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
    df[["diag_1", "diag_2", "diag_3"]] = df[["diag_1", "diag_2", "diag_3"]].fillna(
        "Missing"
    )

    inconnus = {
        "admission_type_id": [5, 6, 8],
        "discharge_disposition_id": [18, 25, 26],
        "admission_source_id": [9, 15, 17, 20, 21],
    }
    for col, codes in inconnus.items():
        df[col] = df[col].replace(codes, 0)

    # 4.3 Réduction de la cardinalité
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].apply(groupe_diag)
    df["medical_specialty"] = df["medical_specialty"].apply(groupe_specialite)

    admission_type_map = {1: "Emergency", 2: "Urgent", 3: "Elective", 0: "Unknown"}
    df["admission_type"] = (
        df["admission_type_id"].map(admission_type_map).fillna("Unknown")
    )

    admission_source_map = {
        7: "Emergency",
        1: "Referral",
        2: "Referral",
        3: "Referral",
        4: "Transfer",
        5: "Transfer",
        6: "Transfer",
        10: "Transfer",
        18: "Transfer",
        22: "Transfer",
        25: "Transfer",
        26: "Transfer",
        0: "Unknown",
    }
    df["admission_source"] = (
        df["admission_source_id"].map(admission_source_map).fillna("Unknown")
    )

    discharge_map = {
        1: "Home",
        6: "Home_health",
        8: "Home_health",
        3: "SNF",
        22: "Rehab",
        13: "Hospice",
        14: "Hospice",
        7: "Left_AMA",
        0: "Unknown",
    }
    df["discharge_disposition"] = (
        df["discharge_disposition_id"].map(discharge_map).fillna("Other_transfer")
    )
    df = df.drop(
        columns=["admission_type_id", "admission_source_id", "discharge_disposition_id"]
    )

    # 4.4 Feature engineering (sur valeurs brutes)
    df["deja_hospitalise"] = (df["number_inpatient"] > 0).astype(int)
    df["total_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    meds = df.loc[:, "metformin":"metformin-pioglitazone"].columns
    df["nb_med_changes"] = df[meds].isin(["Up", "Down"]).sum(axis=1)
    df["nb_meds_diabete"] = (df[meds] != "No").sum(axis=1)
    df = df.drop(columns=[m for m in meds if m not in MEDS_GARDER])

    # Âge : tranche -> milieu de la tranche (en années)
    age_map = {f"[{10 * i}-{10 * (i + 1)})": 10 * i + 5 for i in range(10)}
    df["age"] = df["age"].map(age_map)

    # Cible binaire (readmitted supprimée ensuite : fuite de données sinon)
    df["readmit_30"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # 4.5 Transformations : plafonnement et log1p
    df["number_diagnoses"] = df["number_diagnoses"].clip(upper=9)
    for col in [
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "total_visits",
    ]:
        df[col] = np.log1p(df[col])

    return df


def prepare_data(path=DEFAULT_DATA_PATH, n_splits=5, random_state=RANDOM_STATE):
    """
    Charge le CSV, nettoie les données et sépare train/test PAR PATIENT
    (StratifiedGroupKFold : un patient n'est jamais dans les deux jeux).

    Args:
        path (str): chemin du fichier diabetic_data.csv.
        n_splits (int): 5 plis -> test = ~20 % des données.
        random_state (int): graine pour la reproductibilité.
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # '?' = valeur manquante ; keep_default_na=False garde "None" (test non fait)
    df = pd.read_csv(path, na_values="?", keep_default_na=False, low_memory=False)
    df = clean_data(df)

    X = df.drop(columns=["readmit_30", "encounter_id", "patient_nbr"])
    y = df["readmit_30"]
    groups = df["patient_nbr"]

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. Entraînement
# ---------------------------------------------------------------------------
def build_pipeline(
    n_estimators=200,
    min_samples_leaf=20,
    max_features="sqrt",
    random_state=RANDOM_STATE,
):
    """Construit le Pipeline : ColumnTransformer + Random Forest équilibré."""
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            (
                "cat",
                OneHotEncoder(
                    drop="if_binary", handle_unknown="ignore", sparse_output=False
                ),
                cat_features,
            ),
            ("bin", "passthrough", bin_features),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def train_model(X_train, y_train, **params):
    """
    Entraîne le Pipeline complet sur le jeu d'entraînement.

    Args:
        X_train (pd.DataFrame), y_train (pd.Series): données d'entraînement.
        **params: hyperparamètres optionnels du Random Forest.
    Returns:
        sklearn.pipeline.Pipeline: modèle entraîné.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train et y_train n'ont pas la même longueur.")
    model = build_pipeline(**params)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 3. Évaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, verbose=True):
    """
    Évalue le modèle sur le jeu de test.

    Returns:
        dict: Accuracy, Recall, Precision, F1, ROC-AUC, PR-AUC.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": average_precision_score(y_test, y_proba),
    }
    if verbose:
        print("=== Évaluation sur le jeu de test ===")
        for nom, val in metrics.items():
            print(f"{nom:10s} : {val:.3f}")
        print("Matrice de confusion [[TN FP] [FN TP]] :")
        print(confusion_matrix(y_test, y_pred))
    return metrics


# ---------------------------------------------------------------------------
# 4. Sauvegarde / chargement
# ---------------------------------------------------------------------------
def save_model(model, path=DEFAULT_MODEL_PATH):
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")
    return path


def load_model(path=DEFAULT_MODEL_PATH):
    """Charge un modèle sauvegardé avec joblib."""
    model = joblib.load(path)
    print(f"Modèle chargé : {path}")
    return model


# ---------------------------------------------------------------------------
# 5. Prédiction
# ---------------------------------------------------------------------------
def predict_model(model, X):
    """
    Effectue des prédictions avec le modèle entraîné.

    Args:
        model: modèle entraîné.
        X: données à prédire.

    Returns:
        predictions: classes prédites (0 ou 1).
        probabilities: probabilités de la classe 1.
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return predictions, probabilities
