import sys

import pytest

from src import create_split_dataset as module


class FauxLoader:
    """Faux DatasetLoader : il note ce qu'on lui demande, sans rien copier."""

    appels = []

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def split_and_save_dataset(self, output_base_dir, test_size, random_state):
        FauxLoader.appels.append(
            {
                "data_dir": self.data_dir,
                "output_base_dir": output_base_dir,
                "test_size": test_size,
                "random_state": random_state,
            }
        )


@pytest.fixture(autouse=True)
def faux_loader(monkeypatch):
    FauxLoader.appels = []
    monkeypatch.setattr(module, "DatasetLoader", FauxLoader)


def test_dossier_source_absent_leve_une_erreur(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.create_frozen_split(tmp_path / "inexistant", tmp_path / "sortie")


def test_dossier_source_absent_ne_lance_pas_le_split(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.create_frozen_split(tmp_path / "inexistant", tmp_path / "sortie")
    assert FauxLoader.appels == []


def test_create_frozen_split_transmet_les_bons_parametres(tmp_path):
    source = tmp_path / "processed"
    source.mkdir()
    sortie = tmp_path / "sortie"

    module.create_frozen_split(source, sortie, test_size=0.3, random_state=7)

    assert len(FauxLoader.appels) == 1
    appel = FauxLoader.appels[0]
    assert appel["data_dir"] == source
    assert appel["output_base_dir"] == sortie
    assert appel["test_size"] == 0.3
    assert appel["random_state"] == 7


def test_create_frozen_split_valeurs_par_defaut(tmp_path):
    source = tmp_path / "processed"
    source.mkdir()

    module.create_frozen_split(source, tmp_path / "sortie")

    appel = FauxLoader.appels[0]
    assert appel["test_size"] == 0.2
    assert appel["random_state"] == 42


def test_create_frozen_split_accepte_des_str(tmp_path):
    source = tmp_path / "processed"
    source.mkdir()

    module.create_frozen_split(str(source), str(tmp_path / "sortie"))

    assert len(FauxLoader.appels) == 1


def test_main_utilise_la_config_par_defaut(monkeypatch):
    recu = {}
    monkeypatch.setattr(sys, "argv", ["create_split_dataset.py"])
    monkeypatch.setattr(
        module, "load_config", lambda chemin: {"models": {"test_size": 0.3, "seed": 7}}
    )
    monkeypatch.setattr(module, "create_frozen_split", lambda **kwargs: recu.update(kwargs))

    module.main()

    assert recu["test_size"] == 0.3
    assert recu["random_state"] == 7
    assert recu["processed_dir"] == module.PROCESSED_DATA_DIR
    assert recu["output_dir"] == module.DATA_DIR


def test_main_les_options_ont_priorite_sur_la_config(monkeypatch):
    recu = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_split_dataset.py",
            "--test-size", "0.5",
            "--seed", "1",
            "--data-dir", "mes_images",
            "--output-dir", "ma_sortie",
        ],
    )
    monkeypatch.setattr(
        module, "load_config", lambda chemin: {"models": {"test_size": 0.3, "seed": 7}}
    )
    monkeypatch.setattr(module, "create_frozen_split", lambda **kwargs: recu.update(kwargs))

    module.main()

    assert recu["test_size"] == 0.5
    assert recu["random_state"] == 1
    assert str(recu["processed_dir"]) == "mes_images"
    assert str(recu["output_dir"]) == "ma_sortie"