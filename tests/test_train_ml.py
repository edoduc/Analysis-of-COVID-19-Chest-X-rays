import sys

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src import train_ml as module


# ---------- Aides ----------

def sauver_features(dossier, decoupage, noms, n=6):
    """Crée de petits fichiers .npy : 3 colonnes par feature, 2 classes."""
    dossier.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for nom in noms:
        np.save(dossier / f"X_{decoupage}_{nom}.npy", rng.random((n, 3)))
    np.save(dossier / f"y_{decoupage}.npy", np.array([0, 1] * (n // 2)))


class FauxMlflow:
    """Remplace mlflow : il note ce qu'on lui envoie, sans rien enregistrer."""

    def __init__(self):
        self.params = {}
        self.metrics = {}
        self.uri = None
        self.experience = None
        self.nom_run = None

    def set_tracking_uri(self, uri):
        self.uri = uri

    def get_tracking_uri(self):
        return self.uri

    def set_experiment(self, nom):
        self.experience = nom

    def start_run(self, run_name=None):
        self.nom_run = run_name
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def log_param(self, cle, valeur):
        self.params[cle] = valeur

    def log_metric(self, cle, valeur):
        self.metrics[cle] = valeur

    def log_artifact(self, *args, **kwargs):
        pass


@pytest.fixture
def faux_mlflow(monkeypatch):
    faux = FauxMlflow()
    for nom in [
        "set_tracking_uri",
        "get_tracking_uri",
        "set_experiment",
        "start_run",
        "log_param",
        "log_metric",
        "log_artifact",
    ]:
        monkeypatch.setattr(module.mlflow, nom, getattr(faux, nom))
    # on évite d'enregistrer un vrai modèle dans MLflow
    monkeypatch.setattr(module.mlflow.sklearn, "log_model", lambda **kwargs: None)
    return faux


# ---------- load_and_combine_features ----------

def test_load_and_combine_features_assemble_les_colonnes(tmp_path):
    sauver_features(tmp_path, "train", ["stats", "hog"])
    x, y = module.load_and_combine_features(tmp_path, "train", ["stats", "hog"])
    assert x.shape == (6, 6)  # 3 + 3 colonnes
    assert y.tolist() == [0, 1, 0, 1, 0, 1]


def test_load_and_combine_features_une_seule_feature(tmp_path):
    sauver_features(tmp_path, "test", ["lbp"])
    x, _ = module.load_and_combine_features(tmp_path, "test", ["lbp"])
    assert x.shape == (6, 3)


def test_load_and_combine_features_fichier_feature_absent(tmp_path):
    sauver_features(tmp_path, "train", ["stats"])
    with pytest.raises(FileNotFoundError):
        module.load_and_combine_features(tmp_path, "train", ["stats", "hog"])


def test_load_and_combine_features_labels_absents(tmp_path):
    np.save(tmp_path / "X_train_stats.npy", np.ones((4, 3)))
    with pytest.raises(FileNotFoundError):
        module.load_and_combine_features(tmp_path, "train", ["stats"])


# ---------- build_estimator ----------

def test_build_estimator_svm():
    modele = module.build_estimator("svm", {})
    assert isinstance(modele, SVC)
    assert modele.kernel == "rbf"
    assert modele.probability is True


def test_build_estimator_rf_avec_config():
    modele = module.build_estimator("rf", {"rf": {"n_estimators": 7}})
    assert isinstance(modele, RandomForestClassifier)
    assert modele.n_estimators == 7


def test_build_estimator_xgb_valeurs_par_defaut():
    modele = module.build_estimator("xgb", {})
    assert isinstance(modele, XGBClassifier)
    assert modele.n_estimators == 150
    assert modele.max_depth == 4


def test_build_estimator_nom_en_majuscules():
    assert isinstance(module.build_estimator("RF", {}), RandomForestClassifier)


def test_build_estimator_modele_inconnu():
    with pytest.raises(ValueError):
        module.build_estimator("inconnu", {})


# ---------- build_pipeline ----------

def test_build_pipeline_avec_scaler():
    pipeline = module.build_pipeline("rf", {}, use_scaler=True)
    assert isinstance(pipeline, Pipeline)
    assert [nom for nom, _ in pipeline.steps] == ["scaler", "classifier"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)


def test_build_pipeline_sans_scaler():
    pipeline = module.build_pipeline("rf", {}, use_scaler=False)
    assert [nom for nom, _ in pipeline.steps] == ["classifier"]


def test_build_pipeline_svm_a_toujours_un_scaler():
    pipeline = module.build_pipeline("svm", {}, use_scaler=False)
    assert "scaler" in pipeline.named_steps


# ---------- train ----------

@pytest.fixture
def environnement_train(tmp_path, monkeypatch, faux_mlflow):
    """Dossiers temporaires, config vide et adresse MLflow factice."""
    features = tmp_path / "features"
    modeles = tmp_path / "modeles"
    sauver_features(features, "train", ["stats", "hog"], n=10)
    sauver_features(features, "test", ["stats", "hog"], n=6)
    monkeypatch.setattr(module, "FEATURES_DIR", features)
    monkeypatch.setattr(module, "MODELS_DIR", modeles)
    monkeypatch.setattr(module, "load_config", lambda chemin: {})
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "fichier_factice")
    return {"modeles": modeles, "mlflow": faux_mlflow}


def test_train_sans_adresse_mlflow_leve_une_erreur(environnement_train, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    with pytest.raises(ValueError):
        module.train(model_name="rf", features=["stats", "hog"])


def test_train_sauvegarde_un_modele_utilisable(environnement_train):
    module.train(model_name="rf", features=["stats", "hog"])

    chemin = environnement_train["modeles"] / "rf_stats_hog.joblib"
    assert chemin.exists()
    pipeline = joblib.load(chemin)
    assert len(pipeline.predict(np.random.default_rng(1).random((4, 6)))) == 4


def test_train_enregistre_les_parametres_et_metriques(environnement_train):
    module.train(model_name="rf", features=["stats", "hog"], exp_name="MonExperience")

    faux = environnement_train["mlflow"]
    assert faux.experience == "MonExperience"
    assert faux.nom_run == "rf_stats_hog"
    assert faux.params["model_type"] == "rf"
    assert faux.params["feature_vector_size"] == 6
    assert faux.params["use_scaler"] is True
    for metrique in ["test_accuracy", "test_f1_macro", "test_f1_weighted"]:
        assert 0.0 <= faux.metrics[metrique] <= 1.0


def test_train_sans_scaler(environnement_train):
    module.train(model_name="rf", features=["stats", "hog"], use_scaler=False)

    pipeline = joblib.load(environnement_train["modeles"] / "rf_stats_hog.joblib")
    assert "scaler" not in pipeline.named_steps


def test_train_features_manquantes(environnement_train):
    with pytest.raises(FileNotFoundError):
        module.train(model_name="rf", features=["lbp"])


# ---------- main ----------

def test_main_valeurs_par_defaut(monkeypatch):
    recu = {}
    monkeypatch.setattr(sys, "argv", ["train_ml.py"])
    monkeypatch.setattr(module, "train", lambda **kwargs: recu.update(kwargs))

    module.main()

    assert recu["model_name"] == "rf"
    assert recu["features"] == ["stats", "hog"]
    assert recu["exp_name"] == "XRay_Classical_ML"
    assert recu["use_scaler"] is True
    assert recu["config_path"] is None


def test_main_transmet_les_options(monkeypatch):
    recu = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_ml.py", "--model", "svm", "--features", "lbp", "--no-scale", "--exp_name", "Essai"],
    )
    monkeypatch.setattr(module, "train", lambda **kwargs: recu.update(kwargs))

    module.main()

    assert recu["model_name"] == "svm"
    assert recu["features"] == ["lbp"]
    assert recu["use_scaler"] is False
    assert recu["exp_name"] == "Essai"


def test_main_modele_invalide_quitte(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_ml.py", "--model", "inconnu"])
    with pytest.raises(SystemExit):
        module.main()