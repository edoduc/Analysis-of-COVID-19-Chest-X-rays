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
uv sync
```

`uv sync` télécharge Python 3.12 si nécessaire, crée le dossier `.venv` et installe
les versions exactes listées dans `uv.lock`. Il n'est pas nécessaire d'activer
l'environnement manuellement.

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
3. Cliquer sur **Prédire avec ResNet-50** : la classe prédite, la confiance et les
   probabilités s'affichent, comparées au label attendu.
4. Cliquer sur **Interpréter avec Grad-CAM** : les zones ayant influencé la décision
   sont visualisées.

---

## Guide Développeur / Data Scientist

Pour les développeurs et les data scientists souhaitant reproduire les entraînements de modèles, tester les différentes architectures de Machine Learning ou ajuster les hyperparamètres, veuillez consulter notre [Guide d'entraînement et de reproduction (DEVELOPER.md)](./DEVELOPER.md).
