# Guide d'entraînement et reproduction (Data Science)

Ce guide s'adresse aux développeurs et data scientists souhaitant reproduire l'entraînement de la suite Machine Learning ou modifier les hyperparamètres.

---

## 1. Configurations (`config.json`)

Le fichier `config.json` à la racine est l'unique source de vérité pour :
- **Prétraitement :** Les résolutions de redimensionnement (`image_size`, `model_image_size`).
- **Feature Extraction :** Paramètres pour LBP (`radius`, `method`) et HOG (`orientations`, `pixels_per_cell`, etc.).
- **Modèles :** Hyperparamètres d'entraînement pour SVM, Random Forest (`rf`), et XGBoost (`xgb`).

---

## 2. Pipeline de Reproduction (Fast-Forward)

Pour exécuter et reproduire l'entraînement des modèles de machine learning classiques, exécutez les commandes suivantes dans l'ordre:

### Étape 1 : Prétraitement des images brutes
Applique le pipeline de masquage, recadrage, ajustement du contraste (CLAHE) et sauvegarde dans `data/processed` :
```bash
python src/preprocessing.py
```

### Étape 2 : Split figé (`data/splits.csv`)
Le découpage est défini **uniquement** par `data/splits.csv` (colonnes
`split,class_name,image_path`, chemins relatifs à `data/processed`), versionné avec DVC :
train 70 % / validation 10 % / test 20 % (le test set est celui du ResNet-50).
```bash
uv run dvc pull data/splits.csv.dvc
```
- Tout accès aux données passe par `src.data_loader.DatasetLoader`
  (`load_split` pour le ML, `torch_dataset` pour le DL). Ne jamais re-splitter ailleurs.
- `DatasetLoader.fingerprint()` refuse de tourner si `splits.csv` ne correspond pas à
  `data/splits.csv.dvc` ; chaque run MLflow porte le tag `split_manifest_md5`.

### Étape 3 : Extraction des caractéristiques ML
Les features sont versionnées avec DVC : `uv run dvc pull data/features.dvc` suffit.
Pour les recalculer (stats / LBP / HOG, en 299×299 natif, une matrice par split :
`X_train_*`, `X_validation_*`, `X_test_*`) :
```bash
uv run python -m src.feature_extraction_ml
uv run dvc add data/features
```
`train_ml.py` et `evaluate.py` vérifient que les labels `y_<split>.npy` correspondent
ligne à ligne au split de `splits.csv`, et refusent sinon. Entraînement sur `train`,
réglage sur `validation`, score final sur `test`.

### Étape 4 : Entraînement des modèles
Entraînez un modèle spécifique sur les caractéristiques de votre choix :
```bash
# Random Forest
python src/train_ml.py --model rf --features stats hog

# Support Vector Machine (SVM)
python src/train_ml.py --model svm --features stats lbp hog

# XGBoost
python src/train_ml.py --model xgb --features stats hog
```

### Étape 5 : Exécuter l'étude d'ablation globale
Pour lancer l'entraînement complet de toutes les combinaisons possibles de caractéristiques et de modèles et les enregistrer dans MLflow :
```bash
python src/run_experiments.py
```

---

## 3. Évaluation des Modèles

Une fois entraînés, évaluez les performances globales des modèles et générez les matrices de confusion sous `reports/` :

* **Modèle Machine Learning (ex. Random Forest) :**
  ```bash
  python src/evaluate.py --type ml --filename rf_stats_hog.joblib --features stats hog
  ```

* **Modèle Deep Learning (PyTorch ResNet-50) :**
  ```bash
  python src/evaluate.py --type deep --filename resnet50_best.pth
  ```

Les modèles deep sont toujours évalués sur le split `test` du manifeste (aucun repli
sur un split aléatoire).

* **Modèle Deep Learning Custom CNN (from scratch) :**
  ```bash
  uv run python src/evaluate.py --type deep --arch custom_cnn --filename custom_cnn/best_checkpoint.pt
  ```


---

## 4. Entraîner un modèle Deep Learning (Colab) sur le split figé

Entraînez sur `train`, faites l'early stopping sur `validation`, n'utilisez jamais `test`.
Dans Colab, après avoir copié `data/processed` et `data/splits.csv` :

```python
import hashlib
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

PROCESSED = Path("data/processed")
splits = pd.read_csv("data/splits.csv")

def split_dataset(split, transform):
    folder = ImageFolder(PROCESSED, transform=transform)
    index = {Path(p).relative_to(PROCESSED).as_posix(): i for i, (p, _) in enumerate(folder.samples)}
    paths = sorted(splits.loc[splits["split"] == split, "image_path"])
    return Subset(folder, [index[p] for p in paths])

train_ds = split_dataset("train", train_transforms)
val_ds = split_dataset("validation", eval_transforms)

# Reproductibilité : même ordre de mélange à chaque run, validation non mélangée.
torch.manual_seed(42)
g = torch.Generator().manual_seed(42)
dataloader_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
dataloader_val = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Vérifie que splits.csv est bien la version commitée dans DVC.
SPLIT_MD5 = hashlib.md5(open("data/splits.csv", "rb").read()).hexdigest()
```

Enregistrez la traçabilité dans le checkpoint, à côté de `model_state_dict` :
`split_manifest_md5`, `classes`, `img_size`, `seed`. `src/register_deep_models.py` refuse
un checkpoint dont `split_manifest_md5` ne correspond pas au `splits.csv` courant.

Dans le dépôt : `DatasetLoader().torch_dataset("train", transform)`.
