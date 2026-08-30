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

Pour exécuter et reproduire l'entraînement des modèles de machine learning classiques, exécutez les commandes suivantes dans l'ordre (en utilisant `uv` ou `python` selon votre installation) :

### Étape 1 : Prétraitement des images brutes
Applique le pipeline de masquage, recadrage, ajustement du contraste (CLAHE) et sauvegarde dans `data/processed` :
```bash
uv run python src/preprocessing.py
```

### Étape 2 : Séparation stratifiée du dataset
Sépare les images traitées en répertoires `train` et `test` (80% / 20% stratifié) :
```bash
uv run python src/create_split_dataset.py
```

### Étape 3 : Extraction des caractéristiques ML
Calcule et enregistre de manière indépendante les matrices de caractéristiques statistiques, LBP et HOG :
```bash
uv run python src/feature_extraction_ml.py
```

### Étape 4 : Entraînement des modèles
Entraînez un modèle spécifique sur les caractéristiques de votre choix :
```bash
# Random Forest
uv run python src/train_ml.py --model rf --features stats hog

# Support Vector Machine (SVM)
uv run python src/train_ml.py --model svm --features stats lbp hog

# XGBoost
uv run python src/train_ml.py --model xgb --features stats hog
```

### Étape 5 : Exécuter l'étude d'ablation globale
Pour lancer l'entraînement complet de toutes les combinaisons possibles de caractéristiques et de modèles et les enregistrer dans MLflow :
```bash
uv run python src/run_experiments.py
```

---

## 3. Évaluation des Modèles

Une fois entraînés, évaluez les performances globales des modèles et générez les matrices de confusion sous `reports/` :

* **Modèle Machine Learning (ex. Random Forest) :**
  ```bash
  uv run python src/evaluate.py --type ml --filename rf_stats_hog.joblib --features stats hog
  ```

* **Modèle Deep Learning (PyTorch ResNet-50) :**
  ```bash
  uv run python src/evaluate.py --type deep --filename resnet50_best.pth
  ```
