import sys

import cv2
import numpy as np
import pytest

from src import feature_extraction_ml as module


# ---------- Aides pour fabriquer de fausses images ----------

def image_aleatoire(taille=(32, 32), graine=0):
    """Image en niveaux de gris, pixels entre 1 et 254 (aucun pixel noir)."""
    rng = np.random.default_rng(graine)
    return rng.integers(1, 255, size=taille, dtype=np.uint8)


def creer_dossier_images(dossier, nombre, graine_depart=0):
    dossier.mkdir(parents=True, exist_ok=True)
    for i in range(nombre):
        cv2.imwrite(str(dossier / f"img{i}.png"), image_aleatoire(graine=graine_depart + i))


# ---------- extract_stat_features ----------

def test_stat_features_forme_et_type():
    resultat = module.extract_stat_features(image_aleatoire())
    assert resultat.shape == (15,)
    assert resultat.dtype == np.float32
    assert not np.isnan(resultat).any()


def test_stat_features_premieres_valeurs():
    image = image_aleatoire()
    resultat = module.extract_stat_features(image)
    assert resultat[0] == pytest.approx(image.mean(), rel=1e-4)  # moyenne
    assert resultat[2] == image.min()  # minimum
    assert resultat[3] == image.max()  # maximum


def test_stat_features_image_noire_renvoie_des_zeros():
    resultat = module.extract_stat_features(np.zeros((32, 32), dtype=np.uint8))
    assert resultat.shape == (15,)
    assert not resultat.any()


# ---------- extract_lbp_features ----------

def test_lbp_features_forme_par_defaut_et_somme_a_un():
    resultat = module.extract_lbp_features(image_aleatoire())
    assert resultat.shape == (26,)  # 8 * 3 + 2
    assert resultat.dtype == np.float32
    assert resultat.sum() == pytest.approx(1.0, abs=1e-3)


def test_lbp_features_rayon_personnalise():
    resultat = module.extract_lbp_features(image_aleatoire(), radius=1)
    assert resultat.shape == (10,)  # 8 * 1 + 2


def test_lbp_features_image_noire_renvoie_des_zeros():
    resultat = module.extract_lbp_features(np.zeros((32, 32), dtype=np.uint8))
    assert resultat.shape == (26,)
    assert not resultat.any()


# ---------- extract_hog_features ----------

def test_hog_features_forme_par_defaut():
    resultat = module.extract_hog_features(image_aleatoire((160, 160)))
    assert resultat.shape == (576,)
    assert resultat.dtype == np.float32


def test_hog_features_redimensionne_une_image_plus_petite():
    resultat = module.extract_hog_features(image_aleatoire((64, 64)))
    assert resultat.shape == (576,)  # même taille que pour une image 160x160


def test_hog_features_parametres_personnalises():
    resultat = module.extract_hog_features(
        image_aleatoire((64, 64)), hog_image_size=(64, 64)
    )
    assert resultat.shape == (36,)


# ---------- extract_features_from_images ----------

def test_extract_features_from_images_configs_vides():
    images = [image_aleatoire(graine=i) for i in range(3)]
    stats, lbp, hog = module.extract_features_from_images(images, "test", {}, {})
    assert stats.shape == (3, 15)
    assert lbp.shape == (3, 26)
    assert hog.shape == (3, 576)


def test_extract_features_from_images_configs_personnalisees():
    images = [image_aleatoire(graine=i) for i in range(3)]
    stats, lbp, hog = module.extract_features_from_images(
        images, "test", {"radius": 1}, {"hog_image_size": [64, 64]}
    )
    assert lbp.shape == (3, 10)
    assert hog.shape == (3, 36)


# ---------- save_features ----------

def test_save_features_cree_les_quatre_fichiers(tmp_path):
    sortie = tmp_path / "nouveau_dossier"
    module.save_features(
        sortie,
        "train",
        np.ones((2, 15)),
        np.ones((2, 26)),
        np.ones((2, 576)),
        np.array([0, 1]),
    )
    for nom in ["X_train_stats.npy", "X_train_lbp.npy", "X_train_hog.npy", "y_train.npy"]:
        assert (sortie / nom).exists()
    assert np.load(sortie / "y_train.npy").tolist() == [0, 1]


# ---------- run_feature_extraction ----------

@pytest.fixture
def config_vide(monkeypatch):
    """Évite de lire ton vrai config.json : on utilise les réglages par défaut."""
    monkeypatch.setattr(module, "load_config", lambda chemin: {})


def test_run_feature_extraction_avec_train_et_test_existants(tmp_path, config_vide):
    for decoupage in ["train", "test"]:
        for classe in ["COVID", "Normal"]:
            creer_dossier_images(tmp_path / decoupage / classe, 2)
    sortie = tmp_path / "features"

    module.run_feature_extraction(
        train_dir=tmp_path / "train", test_dir=tmp_path / "test", output_dir=sortie
    )

    assert np.load(sortie / "X_train_stats.npy").shape == (4, 15)
    assert np.load(sortie / "X_test_stats.npy").shape == (4, 15)
    assert np.load(sortie / "X_train_hog.npy").shape == (4, 576)
    assert len(np.load(sortie / "y_train.npy")) == 4
    assert (sortie / "classes.npy").exists()


def test_run_feature_extraction_sans_train_test_decoupe_le_dossier_processed(
    tmp_path, config_vide
):
    processed = tmp_path / "processed"
    creer_dossier_images(processed / "COVID", 5)
    creer_dossier_images(processed / "Normal", 5, graine_depart=100)
    sortie = tmp_path / "features"

    module.run_feature_extraction(
        train_dir=tmp_path / "pas_de_train",
        test_dir=tmp_path / "pas_de_test",
        processed_dir=processed,
        output_dir=sortie,
    )

    assert np.load(sortie / "X_train_stats.npy").shape == (8, 15)
    assert np.load(sortie / "X_test_stats.npy").shape == (2, 15)
    assert np.load(sortie / "classes.npy").tolist() == ["COVID", "Normal"]


# ---------- main ----------

def test_main_transmet_les_options(monkeypatch):
    recu = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["feature_extraction_ml.py", "--output-dir", "sortie", "--config", "c.json"],
    )
    monkeypatch.setattr(module, "run_feature_extraction", lambda **kwargs: recu.update(kwargs))

    module.main()

    assert recu["output_dir"] == "sortie"
    assert recu["config_path"] == "c.json"
    assert recu["train_dir"] is None
    assert recu["test_dir"] is None