import cv2
import numpy as np

from src.data_loader import DatasetLoader


# ---------- Aides pour fabriquer de fausses images ----------

def creer_images(dossier, noms, taille=(20, 20)):
    dossier.mkdir(parents=True, exist_ok=True)
    for i, nom in enumerate(noms):
        image = np.full(taille, i * 10, dtype=np.uint8)
        cv2.imwrite(str(dossier / nom), image)


def creer_dataset(racine, classes):
    """classes = {"COVID": 5, "Normal": 5} -> crée 5 images dans chaque dossier."""
    for nom_classe, nombre in classes.items():
        creer_images(racine / nom_classe, [f"img{i}.png" for i in range(nombre)])


# ---------- get_classes ----------

def test_get_classes_renvoie_les_dossiers_tries(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "fichier.txt").write_text("pas un dossier")
    loader = DatasetLoader(tmp_path)
    assert loader.get_classes() == ["a", "b"]


def test_get_classes_dossier_absent(tmp_path):
    loader = DatasetLoader(tmp_path / "inexistant")
    assert loader.get_classes() == []


# ---------- iter_image_paths ----------

def test_iter_image_paths_garde_seulement_les_images_et_trie(tmp_path):
    dossier = tmp_path / "COVID"
    creer_images(dossier, ["img2.png", "img1.png"])
    (dossier / "notes.txt").write_text("je ne suis pas une image")

    loader = DatasetLoader(tmp_path)
    resultat = [(p.name, cls) for p, cls in loader.iter_image_paths()]

    assert resultat == [("img1.png", "COVID"), ("img2.png", "COVID")]


# ---------- read_single_image ----------

def test_read_single_image_redimensionne(tmp_path):
    creer_images(tmp_path, ["a.png"], taille=(20, 20))
    loader = DatasetLoader(tmp_path, image_size=(8, 8))
    image = loader.read_single_image(tmp_path / "a.png")
    assert image.shape == (8, 8)


def test_read_single_image_illisible_renvoie_none(tmp_path):
    faux = tmp_path / "casse.png"
    faux.write_text("ceci n'est pas une image")
    loader = DatasetLoader(tmp_path, image_size=(8, 8))
    assert loader.read_single_image(faux) is None
    assert loader.read_single_image(tmp_path / "inexistant.png") is None


# ---------- load_dataset ----------

def test_load_dataset_images_et_labels(tmp_path):
    creer_dataset(tmp_path, {"COVID": 2, "Normal": 2})
    loader = DatasetLoader(tmp_path, image_size=(8, 8))
    x, y = loader.load_dataset()
    assert x.shape == (4, 8, 8)
    assert len(y) == 4
    assert set(y) == {0, 1}


def test_load_dataset_dossier_vide(tmp_path):
    loader = DatasetLoader(tmp_path, image_size=(8, 8))
    x, y = loader.load_dataset()
    assert x.shape == (0, 8, 8)
    assert y.size == 0


# ---------- get_train_test_split ----------

def test_get_train_test_split_tailles(tmp_path):
    creer_dataset(tmp_path, {"COVID": 5, "Normal": 5})
    loader = DatasetLoader(tmp_path, image_size=(8, 8))
    x_train, x_test, y_train, y_test, encoder = loader.get_train_test_split(
        test_size=0.2, random_state=42
    )
    assert len(x_train) == 8
    assert len(x_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
    assert list(encoder.classes_) == ["COVID", "Normal"]


# ---------- split_and_save_dataset ----------

def test_split_and_save_dataset_copie_les_fichiers(tmp_path):
    source = tmp_path / "source"
    sortie = tmp_path / "sortie"
    creer_dataset(source, {"COVID": 5, "Normal": 5})

    DatasetLoader(source).split_and_save_dataset(sortie, test_size=0.2, random_state=42)

    for classe in ["COVID", "Normal"]:
        train = {p.name for p in (sortie / "train" / classe).iterdir()}
        test = {p.name for p in (sortie / "test" / classe).iterdir()}
        assert len(train) == 4
        assert len(test) == 1
        assert train.isdisjoint(test)  # aucune image dans les deux groupes
        # les originaux sont copiés, pas déplacés
        assert len(list((source / classe).iterdir())) == 5


def test_split_and_save_dataset_classe_vide_ignoree(tmp_path):
    source = tmp_path / "source"
    sortie = tmp_path / "sortie"
    creer_dataset(source, {"COVID": 5})
    (source / "Vide").mkdir()

    DatasetLoader(source).split_and_save_dataset(sortie)

    assert (sortie / "train" / "COVID").is_dir()
    assert not (sortie / "train" / "Vide").exists()


def test_split_and_save_dataset_est_reproductible(tmp_path):
    source = tmp_path / "source"
    creer_dataset(source, {"COVID": 10})

    DatasetLoader(source).split_and_save_dataset(tmp_path / "sortie1", random_state=42)
    DatasetLoader(source).split_and_save_dataset(tmp_path / "sortie2", random_state=42)

    test1 = sorted(p.name for p in (tmp_path / "sortie1" / "test" / "COVID").iterdir())
    test2 = sorted(p.name for p in (tmp_path / "sortie2" / "test" / "COVID").iterdir())
    assert test1 == test2