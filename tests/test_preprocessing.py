import sys

import cv2
import numpy as np
import pytest

from src import preprocessing as module


# ---------- Aides pour fabriquer de fausses images ----------

def ecrire_image(chemin, graine=0, taille=(40, 40)):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(graine)
    cv2.imwrite(str(chemin), rng.integers(1, 255, size=taille, dtype=np.uint8))


def ecrire_masque(chemin, taille=(40, 40)):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(chemin), np.full(taille, 255, dtype=np.uint8))


def image_avec_bloc():
    """Image noire 40x30 avec un bloc de pixels non nuls au milieu."""
    rng = np.random.default_rng(0)
    image = np.zeros((40, 30), dtype=np.float32)
    image[10:30, 5:25] = rng.integers(50, 200, size=(20, 20))
    return image


# ---------- load_image_from_path ----------

def test_load_image_redimensionne_et_float32(tmp_path):
    ecrire_image(tmp_path / "a.png", taille=(20, 30))
    image = module.load_image_from_path(tmp_path / "a.png", size=10)
    assert image.shape == (10, 10)
    assert image.dtype == np.float32


def test_load_image_illisible_leve_une_erreur(tmp_path):
    with pytest.raises(ValueError):
        module.load_image_from_path(tmp_path / "inexistant.png")


# ---------- load_mask_from_path ----------

def test_load_mask_est_binaire(tmp_path):
    masque = np.zeros((20, 20), dtype=np.uint8)
    masque[:, :10] = 255  # moitié gauche blanche
    cv2.imwrite(str(tmp_path / "m.png"), masque)

    resultat = module.load_mask_from_path(tmp_path / "m.png", size=10)

    assert resultat.shape == (10, 10)
    assert set(np.unique(resultat)) <= {0.0, 1.0}
    assert resultat[:, 0].all()  # côté gauche = 1
    assert not resultat[:, -1].any()  # côté droit = 0


def test_load_mask_illisible_leve_une_erreur(tmp_path):
    with pytest.raises(ValueError):
        module.load_mask_from_path(tmp_path / "inexistant.png")


# ---------- crop_black_borders ----------

def test_crop_black_borders_garde_seulement_le_bloc():
    image = np.zeros((10, 10), dtype=np.float32)
    image[3:6, 2:8] = 5
    resultat = module.crop_black_borders(image)
    assert resultat.shape == (3, 6)
    assert (resultat == 5).all()


def test_crop_black_borders_image_toute_noire_inchangee():
    image = np.zeros((10, 10), dtype=np.float32)
    assert module.crop_black_borders(image).shape == (10, 10)


# ---------- add_padding ----------

def test_add_padding_rend_l_image_carree():
    image = np.ones((4, 10), dtype=np.float32)
    resultat = module.add_padding(image)
    assert resultat.shape == (10, 10)
    assert resultat.sum() == image.sum()  # seuls des zéros ont été ajoutés
    assert (resultat[3:7, :] == 1).all()  # l'image est centrée


def test_add_padding_image_deja_carree():
    image = np.ones((6, 6), dtype=np.float32)
    assert module.add_padding(image).shape == (6, 6)


# ---------- apply_clahe_nonzero ----------

def test_clahe_garde_le_fond_noir():
    image = image_avec_bloc()
    resultat = module.apply_clahe_nonzero(image)
    assert resultat.shape == image.shape
    assert resultat.dtype == np.float32
    assert not resultat[image == 0].any()  # le fond reste à zéro


def test_clahe_image_noire_renvoie_des_zeros():
    resultat = module.apply_clahe_nonzero(np.zeros((20, 20), dtype=np.float32))
    assert not resultat.any()


# ---------- preprocess_image ----------

def test_preprocess_image_forme_finale():
    resultat = module.preprocess_image(image_avec_bloc(), target_size=16)
    assert resultat.shape == (16, 16)
    assert resultat.dtype == np.float32


def test_preprocess_image_masque_vide_donne_une_image_noire():
    image = image_avec_bloc()
    masque = np.zeros_like(image)
    resultat = module.preprocess_image(image, mask=masque, target_size=16)
    assert resultat.shape == (16, 16)
    assert not resultat.any()


# ---------- preprocess_dataset ----------

@pytest.fixture(autouse=True)
def config_factice(monkeypatch):
    """Évite de lire ton vrai config.json."""
    monkeypatch.setattr(
        module, "load_config", lambda chemin=None: {"paths": {"raw_data_dir": "data/raw/x"}}
    )


def test_preprocess_dataset_ecrit_les_images_traitees(tmp_path):
    raw = tmp_path / "raw"
    sortie = tmp_path / "sortie"
    ecrire_image(raw / "COVID" / "images" / "a.png")
    ecrire_masque(raw / "COVID" / "masks" / "a.png")
    ecrire_image(raw / "COVID" / "images" / "b.png", graine=1)  # pas de masque

    module.preprocess_dataset(raw, sortie, classes=["COVID", "Normal"], target_size=32)

    resultat = cv2.imread(str(sortie / "COVID" / "a.png"), cv2.IMREAD_GRAYSCALE)
    assert resultat.shape == (32, 32)
    assert not (sortie / "COVID" / "b.png").exists()  # ignorée : masque absent
    assert (sortie / "Normal").is_dir()  # dossier créé même sans images
    assert list((sortie / "Normal").iterdir()) == []


def test_preprocess_dataset_image_corrompue_ne_plante_pas(tmp_path):
    raw = tmp_path / "raw"
    sortie = tmp_path / "sortie"
    (raw / "COVID" / "images").mkdir(parents=True)
    (raw / "COVID" / "images" / "casse.png").write_text("ceci n'est pas une image")
    ecrire_masque(raw / "COVID" / "masks" / "casse.png")

    module.preprocess_dataset(raw, sortie, classes=["COVID"], target_size=32)

    assert not (sortie / "COVID" / "casse.png").exists()


def test_preprocess_dataset_classes_par_defaut(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sortie = tmp_path / "sortie"

    module.preprocess_dataset(raw, sortie)

    assert sorted(p.name for p in sortie.iterdir()) == sorted(module.DEFAULT_CLASSES)


# ---------- main ----------

def test_main_utilise_la_config(monkeypatch):
    recu = {}
    monkeypatch.setattr(sys, "argv", ["preprocessing.py"])
    monkeypatch.setattr(
        module,
        "load_config",
        lambda chemin: {"image_size": 128, "paths": {"raw_data_dir": "data/raw/x"}},
    )
    monkeypatch.setattr(module, "preprocess_dataset", lambda **kwargs: recu.update(kwargs))

    module.main()

    assert recu["target_size"] == 128
    assert recu["raw_dir"] == module.PROJECT_ROOT / "data/raw/x"
    assert recu["output_dir"] == module.PROJECT_ROOT / "data" / "processed"


def test_main_les_options_ont_priorite(monkeypatch):
    recu = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["preprocessing.py", "--size", "64", "--raw-dir", "mes_raw", "--output-dir", "ma_sortie"],
    )
    monkeypatch.setattr(
        module,
        "load_config",
        lambda chemin: {"image_size": 128, "paths": {"raw_data_dir": "data/raw/x"}},
    )
    monkeypatch.setattr(module, "preprocess_dataset", lambda **kwargs: recu.update(kwargs))

    module.main()

    assert recu["target_size"] == 64
    assert recu["raw_dir"] == "mes_raw"
    assert recu["output_dir"] == "ma_sortie"