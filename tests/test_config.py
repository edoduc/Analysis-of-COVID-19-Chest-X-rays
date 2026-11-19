import json

import pytest

from src import config


def test_project_root_contient_src():
    assert (config.PROJECT_ROOT / "src").is_dir()


def test_dossiers_sont_dans_le_projet():
    for chemin in [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.TRAIN_DATA_DIR,
        config.TEST_DATA_DIR,
        config.FEATURES_DIR,
        config.MODELS_DIR,
        config.REPORTS_DIR,
    ]:
        assert config.PROJECT_ROOT in chemin.parents


def test_default_classes():
    assert len(config.DEFAULT_CLASSES) == 4
    assert "COVID" in config.DEFAULT_CLASSES
    assert "Normal" in config.DEFAULT_CLASSES


def test_load_config_fichier_absent_renvoie_defauts(tmp_path):
    cfg = config.load_config(tmp_path / "inexistant.json")
    assert cfg["image_size"] == 299
    assert cfg["model_image_size"] == 224
    assert cfg["models"]["seed"] == 42
    assert cfg["models"]["test_size"] == 0.2


def test_load_config_lit_un_fichier_json(tmp_path):
    fichier = tmp_path / "ma_config.json"
    fichier.write_text(json.dumps({"image_size": 128}), encoding="utf-8")
    assert config.load_config(fichier) == {"image_size": 128}


def test_load_config_json_invalide_leve_une_erreur(tmp_path):
    fichier = tmp_path / "casse.json"
    fichier.write_text("{ ceci n'est pas du json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        config.load_config(fichier)