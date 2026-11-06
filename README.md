# Analyse de radiographies pulmonaires Covid-19

## Contexte

La pandémie de Covid-19 a renforcé le besoin d'outils de diagnostic rapides et
accessibles. Les radiographies pulmonaires constituent une source d'information
complémentaire aux tests PCR pour détecter les atteintes respiratoires.

## Objectif

Classifier automatiquement des radiographies pulmonaires selon quatre pathologies —
**Normal**, **COVID**, **Lung Opacity** et **Viral Pneumonia** — en comparant plusieurs
approches de computer vision, puis en interprétant les décisions du meilleur modèle.

## Méthode

Le projet s'appuie sur le dataset public
[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
(radiographies en niveaux de gris + masques de segmentation pulmonaire). La démarche
comprend :

- un **pré-traitement** des images (masquage pulmonaire, recadrage, redimensionnement,
  amélioration du contraste par CLAHE) ;
- une **approche machine learning** à partir de caractéristiques extraites (statistiques,
  LBP, HOG) avec SVM, KNN puis XGBoost ;
- une **approche deep learning** avec un ResNet-50 (transfer learning) et un CNN entraîné
  from scratch ;
- une analyse d'**interprétabilité** par Grad-CAM.

Le meilleur modèle est **ResNet-50**, qui sert de base à la démonstration.

## Technologies

- **Python 3.12**
- **uv** — gestion de l'environnement et des dépendances
- **PyTorch** / **torchvision** — deep learning (ResNet-50)
- **grad-cam** — interprétabilité
- **scikit-learn** / **XGBoost** — machine learning
- **OpenCV**, **scikit-image**, **Pillow**, **NumPy**, **pandas** — traitement d'images et de données
- **FastAPI** — API d'inférence
- **Streamlit** — application de présentation et de démonstration

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) pour gérer Python et les dépendances.

1. Installer uv (une seule fois) :

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Cloner le dépôt puis installer l'environnement :

```bash
git clone <url-du-depot>
cd Analysis-of-COVID-19-Chest-X-rays
uv sync --locked
```

`uv sync` télécharge Python 3.12 si nécessaire, crée le dossier `.venv` et installe
les versions exactes listées dans `uv.lock`. Il n'est pas nécessaire d'activer
l'environnement manuellement.

Sur un réseau d'entreprise qui intercepte TLS et provoque `UnknownIssuer`, utiliser
le magasin de certificats Windows :

```powershell
uv --system-certs sync --locked
```

> **PyTorch et GPU** : le projet est configuré pour PyTorch avec CUDA 12.6
> (index `pytorch-cu126` dans `pyproject.toml`). Prévoir environ 10 Go d'espace
> libre pour l'installation. Sans GPU NVIDIA, remplacer `cu126` par `cpu` dans
> l'URL de l'index et retirer le suffixe `+cu126` des versions de torch, puis
> relancer `uv lock && uv sync`.

Vérifier l'installation :

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Lancer l'application

```bash
uv run streamlit run streamlit_app/app.py
```

## Utiliser la démonstration

La dernière section de l'application permet de tester le modèle sur une radiographie :

1. **Sélectionner une image** dans l'un des quatre menus déroulants (un par classe).
2. Cliquer sur **Pré-traiter** : l'image pré-traitée s'affiche à côté de l'originale.
3. Cliquer sur **Prédire avec XGBoost** lorsque Streamlit est relié à l'API ; sans API,
  la démonstration locale historique utilise ResNet-50. Sans masque pulmonaire, l'API
  demande une revue humaine au lieu d'émettre une classe.
4. Grad-CAM reste disponible uniquement pour la démonstration locale ResNet-50 ; il ne
  s'applique pas au modèle d'arbres XGBoost servi par l'API.

## API d'inférence

L'API sert par défaut l'artefact XGBoost du projet Data Science,
`models/xgboost/xgb_best.pkl`. Il est chargé sur CPU et ne nécessite pas de nouvel
entraînement :

```bash
uv run uvicorn api.app:app --host 127.0.0.1 --port 8000
```

La documentation interactive est disponible sur `http://127.0.0.1:8000/docs`.
`POST /predict` accepte un fichier `image` PNG/JPEG et un fichier `mask` PNG/JPEG.
Le masque est nécessaire pour une prédiction automatique cohérente avec l'entraînement.
Sans masque, l'API renvoie `review_required`. Avec un masque, une confiance sous le
seuil configurable renvoie également `review_required`.

Le seuil par défaut est `0.5` et les probabilités ne sont **pas calibrées** en phase 1 ;
ce seuil est provisoire et ne constitue pas un seuil clinique. L'API est un démonstrateur
pédagogique, pas un outil de diagnostic.

Pour choisir l'artefact XGBoost ou régler le seuil sous PowerShell :

```powershell
$env:XGBOOST_MODEL_PATH = "models/xgboost/xgb_best.pkl"
$env:XGBOOST_MODEL_VERSION = "xgb-best-ds"
$env:INFERENCE_CONFIDENCE_THRESHOLD = "0.5"
```

Pour relier l'application Streamlit locale à l'API, définir également :

```powershell
$env:INFERENCE_API_URL = "http://127.0.0.1:8000"
uv run streamlit run streamlit_app/app.py
```

## Lancer avec Docker Compose

Docker Desktop doit être installé et démarré. Depuis la racine du dépôt :

```bash
docker compose up --build
```

L'API est exposée sur `http://localhost:8000` et Streamlit sur
`http://localhost:8501`. Le build inclut le pickle XGBoost de 4,5 Mo et les échantillons
de démonstration, mais exclut le dataset complet. L'environnement partagé inclut encore
PyTorch pour la démonstration locale ResNet ; l'image peut donc rester volumineuse malgré
l'inférence API XGBoost sur CPU.

---

## Guide Développeur / Data Scientist

Pour les développeurs et les data scientists souhaitant reproduire les entraînements de modèles, tester les différentes architectures de Machine Learning ou ajuster les hyperparamètres, veuillez consulter notre [Guide d'entraînement et de reproduction (DEVELOPER.md)](./DEVELOPER.md).
