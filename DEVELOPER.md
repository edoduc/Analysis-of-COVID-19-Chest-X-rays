# Guide d'entraînement et reproduction (Data Science)

Ce guide décrit le pipeline de données existant et la chaîne MLOps XGBoost de phase 1.
ResNet-50 reste disponible pour la comparaison Data Science et Grad-CAM, mais n'est pas
le modèle servi par défaut. Le dataset complet n'est pas stocké dans Git ; il doit être
téléchargé conformément à sa licence et placé selon la structure ci-dessous.

---

## 1. Configurations (`config.json`)

Le fichier `config.json` à la racine contient les chemins et hyperparamètres pour :
- **Prétraitement :** Les résolutions de redimensionnement (`image_size`, `model_image_size`).
- **Feature Extraction :** Paramètres pour LBP (`radius`, `method`) et HOG (`orientations`, `pixels_per_cell`, etc.).
- **Modèles :** Hyperparamètres d'entraînement pour SVM, Random Forest (`rf`), et XGBoost (`xgb`).

Les noms et l'ordre des classes sont définis dans `src/constants.py` et doivent rester
identiques dans les labels, l'API et les artefacts du modèle.

---

## 2. Pipeline de Reproduction (Fast-Forward)

Les images sources doivent suivre la structure
`data/raw/COVID-19_Radiography_Dataset/<classe>/{images,masks}`. Exécuter les commandes
depuis la racine du dépôt. Les scripts du paquet `src` s'exécutent avec `python -m`
pour que les imports `src.*` fonctionnent correctement.

### Étape 1 : Prétraitement des images brutes
Applique le pipeline de masquage, recadrage, ajustement du contraste (CLAHE) et sauvegarde dans `data/processed/<classe>` :
```bash
uv run python -m src.preprocessing
```

### Étape 2 : Séparation stratifiée du dataset
Crée `data/splits.csv` avec le split test stratifié global 80/20 des notebooks DS
(seed 42), puis réserve 10% du total en validation à partir des 80% d'entraînement ; les
images ne sont pas recopiées :
```bash
uv run python -m src.create_split_dataset
```
Le split test reste réservé à l'évaluation finale. Le contenu volumineux de `data/` reste
exclu de Git ; ses manifestes DVC sont versionnés séparément. Le remote par défaut
`dagshub` est configuré dans `.dvc/config`.

### Étape 3 : Extraction des caractéristiques ML
Lit le manifeste commun à 299×299, comme les notebooks DS, et enregistre séparément les
matrices STAT (15), LBP (26) et HOG (576) pour train, validation et test. HOG redimensionne
ensuite en interne à 160×160, comme dans le notebook :
```bash
uv run python -m src.feature_extraction_ml
```

### Suivi DVC des données et des features
DVC avec son backend S3 et Prefect sont fournis dans le groupe optionnel `mlops` du projet.
Pour installer ce groupe dans le venv du projet :
```bash
uv sync --group mlops
```

Après le téléchargement du dataset source et la génération des artefacts, leurs manifestes DVC sont créés ainsi :
```bash
uv run --group mlops dvc add data/raw
uv run --group mlops dvc add data/processed
uv run --group mlops dvc add data/splits.csv
uv run --group mlops dvc add data/features
```
Le manifeste `data/raw.dvc` fixe aussi la version exacte du dataset source utilisée par
le prétraitement. Le dataset agrège des images de plusieurs sources; avant de pousser
`data/raw.dvc` vers un remote, vérifier que les droits de toutes ces sources autorisent
leur réhébergement. La fiche Kaggle indique un usage académique/non commercial et
attribue les fichiers à leurs auteurs, sans établir à elle seule un droit général de
redistribution.

Les manifestes `data/raw.dvc`, `data/processed.dvc`, `data/splits.csv.dvc` et
`data/features.dvc`, ainsi que les métadonnées DVC et les règles `.gitignore`, doivent
être versionnés dans Git. DVC n'active pas l'auto-staging. Les credentials DagsHub sont
locaux dans `.dvc/config.local` et ne doivent jamais être commités. Après avoir
configuré le token sur chaque machine et vérifié les droits de redistribution du jeu
brut, récupérer ou envoyer les objets DVC avec :
```bash
uv run --group mlops dvc pull -r dagshub data/raw.dvc data/processed.dvc data/splits.csv.dvc data/features.dvc
uv run --group mlops dvc push -r dagshub data/raw.dvc data/processed.dvc data/splits.csv.dvc data/features.dvc
```

### Orchestration locale avec Prefect
Le flow exécute les scripts existants de prétraitement, split, extraction, suivi DVC,
entraînement MLflow et évaluation sur le test réservé. Il utilise le moteur local de
Prefect, qui démarre automatiquement un serveur temporaire local pour le run ; aucun
serveur permanent n'est à lancer. MLflow est fourni par le groupe `experiments` :
```bash
uv sync --group mlops --group experiments
uv run --group mlops --group experiments python -m src.prefect_pipeline --model xgb --features stats lbp hog
```
Le modèle peut être `rf`, `svm` ou `xgb`. L'option `--push-data` envoie les artefacts
après leur ajout DVC ; sans cette option, le flow ne pousse rien vers le remote. Le flow
ne committe pas les manifestes modifiés : les vérifier et les versionner avec Git après
exécution.

### Étape 4 : Entraînement des modèles
Les modèles classiques utilisent le split validation pour comparer les expériences et
laissent le test intact. Le groupe `experiments` fournit MLflow pour ces scripts :
```bash
# Random Forest
uv run --group experiments python -m src.train_ml --model rf --features stats hog

# Support Vector Machine (SVM)
uv run --group experiments python -m src.train_ml --model svm --features stats lbp hog

# XGBoost
uv run --group experiments python -m src.train_ml --model xgb --features stats hog
```

### Étape 5 : Modèle de service XGBoost
La chaîne MLOps réutilise d'abord l'artefact Data Science optimisé
`models/xgboost/xgb_best.pkl` (617 colonnes STAT+LBP+HOG) : pas de recherche
d'hyperparamètres longue ni de nouvel entraînement ResNet. L'API calcule les features
de chaque image à la volée avec le prétraitement commun.

Pour réentraîner un candidat XGBoost avec les paramètres fixes de `config.json`, sans
RandomizedSearchCV :
```bash
uv run python -m src.feature_extraction_ml
uv run --group experiments python -m src.train_ml --model xgb --features stats lbp hog
```
Le candidat est enregistré dans `models/saved_models/`. Pour le servir, définir
`XGBOOST_MODEL_PATH` comme décrit dans le README. Le script ResNet figé reste optionnel
pour les comparaisons Data Science ; il n'est pas dans la voie MLOps par défaut.

### Étape 6 : Étude d'ablation classique
Le lanceur actuel compare les combinaisons de caractéristiques avec RF et SVM :
```bash
uv run --group experiments python -m src.run_experiments
```

---

## 3. Évaluation des Modèles

Une fois entraînés, évaluez les performances globales des modèles et générez les matrices de confusion sous `reports/` :

* **Modèle Machine Learning (ex. Random Forest) :**
  ```bash
  uv run python -m src.evaluate --type ml --filename rf_stats_hog.joblib --features stats hog
  ```

* **Artefact XGBoost optimisé du projet Data Science :**
  ```bash
  uv run python -m src.feature_extraction_ml
  uv run python -m src.evaluate --type ml --filename xgb_best.pkl --features stats lbp hog
  ```
  Le pickle et les features utilisent les mêmes 617 colonnes nommées dans le même ordre.

* **Modèle Deep Learning (PyTorch ResNet-50) :**
  ```bash
  uv run python -m src.evaluate --type deep --filename resnet50_best.pth
  ```
  L'évaluateur utilise le split `test` du manifeste `data/splits.csv`.

* **Modèle Deep Learning Custom CNN (from scratch) :**
  ```bash
  uv run python -m src.evaluate --type deep --arch custom_cnn --filename cnn_from_scratch.pt
  ```

## Critères d'acceptation de la phase 1

- Le prétraitement et la création du manifeste s'exécutent de façon reproductible ; les
  splits sont disjoints et stratifiés. Le split de test reste réservé à l'évaluation finale.
- Un entraînement candidat termine et journalise dans MLflow les paramètres, le F1 macro,
  le F1 pondéré et le rappel COVID sur validation.
- Un candidat n'est retenu que si son F1 macro et son rappel COVID sur le même split de
  validation ne régressent pas par rapport au modèle de référence. Ces métriques sont
  rapportées sur le test uniquement après sélection du candidat. Cette comparaison est
  manuelle en phase 1 ; son automatisation relève du pipeline CI des phases suivantes.
- La suite de tests passe ; l'API expose `/health` et `/predict`, rejette les fichiers
  invalides, exige le masque pour prédire et applique la règle de confiance ci-dessous.

Le seuil technique initial est `0.5`, défini dans `config.json` et surchargeable par
`INFERENCE_CONFIDENCE_THRESHOLD` : une confiance strictement inférieure impose une revue
humaine ; une confiance égale ou supérieure autorise une prédiction si le masque est
présent. Les probabilités ne sont pas calibrées. Ce seuil ne constitue donc ni une garantie
d'incertitude ni un seuil clinique. Aucun seuil absolu de performance clinique n'est fixé
sans validation par les responsables médicaux ; le prototype n'est pas destiné au diagnostic.

## API et tests de phase 1

```bash
uv run uvicorn api.app:app --host 127.0.0.1 --port 8000
uv run pytest
```

L'API expose `/health` et `/predict` et charge XGBoost depuis `XGBOOST_MODEL_PATH`.
Le champ `model_family` vaut `xgboost` et `confidence_calibrated` vaut `false` en phase 1.
Sans masque, la réponse est immédiatement `review_required` ; avec masque, la classe est
renvoyée seulement si la confiance dépasse le seuil configuré. Grad-CAM reste réservé
au ResNet local. Ce flux est un prototype pédagogique et ne doit pas servir au diagnostic.

Pour construire l'image et lancer l'API avec Streamlit :

```bash
docker compose up --build
```
