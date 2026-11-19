import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # dessine sans ouvrir de fenêtre

import numpy as np
import pytest
import torch
import torch.nn as nn

from src import evaluate as module


# ---------- Faux modèles ----------

class FauxModele:
    def __init__(self):
        self.forme_recue = None

    def predict(self, x):
        self.forme_recue = x.shape
        return np.zeros(len(x), dtype=int)

    def predict_proba(self, x):
        return np.full((len(x), 2), 0.5)


class FauxModeleSansProba:
    def predict(self, x):
        return np.zeros(len(x), dtype=int)


# ---------- resolve_model_path ----------

def test_resolve_model_path_chemin_existant(tmp_path):
    fichier = tmp_path / "modele.joblib"
    fichier.write_text("x")
    assert module.resolve_model_path(str(fichier)) == fichier


def test_resolve_model_path_cherche_dans_les_dossiers_standards(tmp_path, monkeypatch):
    (tmp_path / "mon_modele_de_test.joblib").write_text("x")
    monkeypatch.setattr(module, "MODELS_DIR", tmp_path)
    resultat = module.resolve_model_path("mon_modele_de_test.joblib")
    assert resultat == tmp_path / "mon_modele_de_test.joblib"


def test_resolve_model_path_introuvable_leve_une_erreur():
    with pytest.raises(FileNotFoundError):
        module.resolve_model_path("fichier_qui_nexiste_vraiment_pas.joblib")


# ---------- evaluate_ml_model ----------

def test_evaluate_ml_model_assemble_les_features(tmp_path, monkeypatch):
    np.save(tmp_path / "X_test_stats.npy", np.ones((4, 3)))
    np.save(tmp_path / "X_test_hog.npy", np.ones((4, 2)))
    np.save(tmp_path / "y_test.npy", np.array([0, 1, 0, 1]))
    faux = FauxModele()
    monkeypatch.setattr(module.joblib, "load", lambda chemin: faux)

    y_test, y_pred, y_prob = module.evaluate_ml_model(
        Path("modele.joblib"), tmp_path, ["stats", "hog"]
    )

    assert faux.forme_recue == (4, 5)  # 3 + 2 colonnes mises côte à côte
    assert list(y_test) == [0, 1, 0, 1]
    assert len(y_pred) == 4
    assert y_prob.shape == (4, 2)


def test_evaluate_ml_model_sans_predict_proba(tmp_path, monkeypatch):
    np.save(tmp_path / "X_test_stats.npy", np.ones((4, 3)))
    np.save(tmp_path / "y_test.npy", np.array([0, 1, 0, 1]))
    monkeypatch.setattr(module.joblib, "load", lambda chemin: FauxModeleSansProba())

    _, _, y_prob = module.evaluate_ml_model(Path("m.joblib"), tmp_path, ["stats"])

    assert y_prob is None


# ---------- load_pytorch_checkpoint ----------

def test_load_pytorch_checkpoint_state_dict_simple(tmp_path):
    source = nn.Linear(2, 2)
    chemin = tmp_path / "poids.pth"
    torch.save(source.state_dict(), chemin)

    nouveau = nn.Linear(2, 2)
    resultat = module.load_pytorch_checkpoint(nouveau, chemin, torch.device("cpu"))

    assert torch.equal(resultat.weight, source.weight)
    assert resultat.training is False  # le modèle est en mode évaluation


def test_load_pytorch_checkpoint_format_model_state_dict(tmp_path):
    source = nn.Linear(2, 2)
    chemin = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": source.state_dict()}, chemin)

    nouveau = nn.Linear(2, 2)
    resultat = module.load_pytorch_checkpoint(nouveau, chemin, torch.device("cpu"))

    assert torch.equal(resultat.weight, source.weight)


# ---------- plot_confusion_matrix ----------

def test_plot_confusion_matrix_cree_le_fichier(tmp_path):
    sortie = tmp_path / "sous_dossier" / "cm.png"
    module.plot_confusion_matrix(
        np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), ["A", "B"], sortie
    )
    assert sortie.exists()
    assert sortie.stat().st_size > 0


# ---------- main ----------

def test_main_ml_affiche_la_precision_et_sauve_la_matrice(tmp_path, monkeypatch, capsys):
    y = np.array([0, 1, 2, 3])
    recu = {}
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--type", "ml", "--filename", "modele.joblib"]
    )
    monkeypatch.setattr(module, "resolve_model_path", lambda nom: Path(nom))
    monkeypatch.setattr(module, "FEATURES_DIR", tmp_path)  # pas de classes.npy ici
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(module, "evaluate_ml_model", lambda *a, **k: (y, y, None))
    monkeypatch.setattr(
        module,
        "plot_confusion_matrix",
        lambda y_true, y_pred, noms, chemin: recu.update(noms=list(noms), chemin=chemin),
    )

    module.main()

    affichage = capsys.readouterr().out
    assert "Overall Accuracy: 1.0000" in affichage
    assert recu["noms"] == module.DEFAULT_CLASSES
    assert recu["chemin"] == tmp_path / "reports" / "cm_modele.png"


def test_main_sans_arguments_obligatoires_quitte(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])
    with pytest.raises(SystemExit):
        module.main()