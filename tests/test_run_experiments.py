import subprocess
import sys

import pytest

from src import run_experiments as module


@pytest.fixture
def faux_subprocess(monkeypatch):
    """Remplace subprocess.run : on note les commandes sans rien lancer."""
    commandes = []

    def faux_run(cmd, check=False):
        commandes.append(cmd)

    monkeypatch.setattr(module.subprocess, "run", faux_run)
    return commandes


# ---------- get_all_combinations ----------

def test_combinaisons_de_trois_features():
    resultat = module.get_all_combinations(["stats", "lbp", "hog"])
    assert len(resultat) == 7  # 3 seuls + 3 paires + 1 triplet
    assert ("stats",) in resultat
    assert ("stats", "lbp") in resultat
    assert ("stats", "lbp", "hog") in resultat


def test_combinaisons_ordre_solos_puis_paires():
    resultat = module.get_all_combinations(["a", "b"])
    assert resultat == [("a",), ("b",), ("a", "b")]


def test_combinaisons_liste_vide():
    assert module.get_all_combinations([]) == []


def test_combinaisons_une_seule_feature():
    assert module.get_all_combinations(["a"]) == [("a",)]


# ---------- run_ablation_study ----------

def test_run_ablation_study_script_absent_leve_une_erreur(tmp_path, monkeypatch, faux_subprocess):
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        module.run_ablation_study()
    assert faux_subprocess == []


def test_run_ablation_study_nombre_de_lancements(faux_subprocess):
    module.run_ablation_study(models=["rf", "svm"], features=["stats", "lbp"])
    # 2 modèles x 3 combinaisons (stats, lbp, stats+lbp)
    assert len(faux_subprocess) == 6


def test_run_ablation_study_forme_de_la_commande(faux_subprocess):
    module.run_ablation_study(models=["rf"], features=["stats", "hog"], exp_name="MonTest")
    commande = faux_subprocess[0]
    assert commande[0] == sys.executable
    assert commande[1].endswith("train_ml.py")
    assert commande[2:] == ["--model", "rf", "--features", "stats", "--exp_name", "MonTest"]


def test_run_ablation_study_passe_plusieurs_features(faux_subprocess):
    module.run_ablation_study(models=["rf"], features=["stats", "hog"])
    derniere = faux_subprocess[-1]
    assert "stats" in derniere and "hog" in derniere


def test_run_ablation_study_continue_apres_un_echec(monkeypatch):
    appels = []

    def run_qui_echoue(cmd, check=False):
        appels.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(module.subprocess, "run", run_qui_echoue)

    module.run_ablation_study(models=["rf", "svm"], features=["stats"])

    assert len(appels) == 2  # le 2e modèle est tenté malgré l'échec du 1er


# ---------- main ----------

def test_main_valeurs_par_defaut(monkeypatch):
    recu = {}
    monkeypatch.setattr(sys, "argv", ["run_experiments.py"])
    monkeypatch.setattr(module, "run_ablation_study", lambda **kwargs: recu.update(kwargs))

    module.main()

    assert recu["models"] == ["rf", "svm"]
    assert recu["features"] == ["stats", "lbp", "hog"]
    assert recu["exp_name"] == "XRay_Ablation_Study"


def test_main_transmet_les_options(monkeypatch):
    recu = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiments.py", "--models", "xgb", "--features", "hog", "--exp_name", "Essai"],
    )
    monkeypatch.setattr(module, "run_ablation_study", lambda **kwargs: recu.update(kwargs))

    module.main()

    assert recu["models"] == ["xgb"]
    assert recu["features"] == ["hog"]
    assert recu["exp_name"] == "Essai"