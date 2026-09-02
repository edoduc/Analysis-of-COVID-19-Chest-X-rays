import sys
from pathlib import Path

# Streamlit runs this file directly, so the repository root may not be importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from streamlit_app.sections import demo
from streamlit_app.components import metric_row, section_header, show_figure
from streamlit_app.config import CLASS_NAMES

# Config de la page Streamlit
st.set_page_config(
    page_title="Analyse de radiographies Covid-19",
    page_icon="🏥",
    layout="wide",
)


def main() -> None:
    # Sidebar de navigation sous forme de "slides"
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")

    sections_dict = {
        "1. Présentation du projet": "1.",
        "2. Contexte et Objectifs": "1.1",
        "3. Exploration des données": "2.",
        "4. Pré-traitement": "3.",
        "5. Extraction de caractéristiques": "4.",
        "6. Approche Baseline (SVM)": "5.",
        "7. Approche Ensemble (XGBoost)": "6.",
        "8. Approche CNN - ResNet-50": "7.",
        "9. Custom CNN from scratch": "8.",
        "10. Interprétabilité (Grad-CAM)": "9.",
        "11. Conclusion générale": "10.",
        "12. Démonstration interactive": "11.",
    }

    selected_section_name = st.sidebar.radio(
        "Diapositives",
        list(sections_dict.keys()),
        index=0,
    )
    section = sections_dict[selected_section_name]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size: 0.8rem; color: #9ca3af; text-align: center;'>"
        "Soutenance de Projet — Janvier 2026"
        "</div>",
        unsafe_allow_html=True,
    )

    # Rendu des sections (slides)
    if section.startswith("1.") and not section.startswith("1.1"):
        _, center_column, _ = st.columns([1, 3, 1])
        with center_column:
            _, logo_column, _ = st.columns([2, 1, 2])
            with logo_column:
                show_figure("0-liora_logo.png")
            st.markdown(
                "<div style='text-align:center'>"
                "<p style='color:#6b7280'>Promotion MLE janvier 2026</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div style='text-align:center'>"
                "<h2 style='margin-bottom:0'>Projet Data Science</h2>"
                "<p style='color:#9ca3af'>-----</p>"
                "<h1 style='margin-top:0'>Analyse de radiographies pulmonaires Covid-19</h1>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div style='text-align:center; color:#374151'>"
                "<p style='margin:0'>Alisa MALKA</p>"
                "<p style='margin:0'>Aziz CHENNOUFI</p>"
                "<p style='margin:0'>Edouard DUCLOY</p>"
                "</div>",
                unsafe_allow_html=True,
            )
    elif section.startswith("1.1"):
        section_header(
            1,
            "Introduction",
            "Intelligence artificielle et analyse de radiographies pulmonaires",
        )
        st.subheader("Contexte")
        st.markdown(
            "- **Besoin critique :** Outils de diagnostic rapides, fiables et accessibles durant la pandémie\n"
            "- **Limites de la PCR :** Disponibilité variable, coûts élevés et délais d'analyse parfois longs\n"
            "- **Radiographies 2D :** Alternative complémentaire, rapide et largement disponible dans les hôpitaux\n"
            "- **Rôle de l'IA :** Computer vision et Deep Learning pour extraire des caractéristiques complexes (COVID, sains, autres pneumonies)"
        )

        scope_column, objective_column = st.columns(2)
        with scope_column:
            st.subheader("Périmètre")
            st.markdown(
                "- Dataset public **COVID-19 Radiography Database**\n"
                "- Radiographies pulmonaires 2D en niveaux de gris\n"
                "- Quatre classes : Normal, COVID, Lung Opacity et Viral Pneumonia\n"
                "- Masques pulmonaires pour isoler la région d'intérêt\n"
                "- Comparaison entre machine learning classique et deep learning"
            )
        with objective_column:
            st.subheader("Objectif")
            st.markdown(
                "- **Classification automatique :** Détecter 4 pathologies à partir de radiographies pulmonaires 2D\n"
                "- **Analyse comparative :** Évaluer et comparer le Machine Learning classique et le Deep Learning\n"
                "- **Impact de la segmentation :** Étudier l'influence du masquage pulmonaire sur les performances\n"
                "- **Explicabilité (Grad-CAM) :** Visualiser les régions pulmonaires décisionnelles pour les cliniciens"
            )

        st.subheader("Déroulé de la soutenance")
        st.markdown(
            "- **Exploration et analyse des données** : structure du dataset, classes, dimensions et premières observations\n"
            "- **Pré-traitement** : segmentation pulmonaire, homogénéisation et amélioration du contraste (CLAHE)\n"
            "- **Machine Learning** : baselines avec SVM et KNN, puis classification avancée avec XGBoost\n"
            "- **Deep learning** : comparaison entre un Custom CNN (from scratch) et ResNet-50 (transfer learning)\n"
            "- **Interprétabilité et bilan** : Grad-CAM, comparaison finale des résultats et démonstration interactive"
        )
    elif section.startswith("2."):
        section_header(
            2,
            "Exploration et analyse des données",
            "Vue globale, aperçu des images et principaux enseignements",
        )
        metric_row(
            [
                ("Radiographies", "21 165"),
                ("Classes", "4"),
            ]
        )
        st.markdown(
            "- **Multi-classes (4 pathologies) :** Classification de pneumonies virales, COVID, opacités et cas sains\n"
            "- **Structure :** Radiographies 2D en niveaux de gris + Masques de segmentation pulmonaire associés"
        )

        st.subheader("Aperçu des données")
        st.markdown(
            "Pour chaque classe : image brute, masque de segmentation et image segmentée."
        )
        sample_figures = {
            "Normal": "3.1-sample_data_normal.png",
            "COVID": "3.1-sample_data_covid.png",
            "Lung Opacity": "3.1-sample_data_lung_opacity.png",
            "Viral Pneumonia": "3.1-sample_data_viral_pneumonia.png",
        }
        selected_class = st.selectbox("Classe", list(sample_figures.keys()))
        show_figure(sample_figures[selected_class], caption=f"Échantillon — {selected_class}")

        st.subheader("Un jeu de données déséquilibré")
        text_column, chart_column = st.columns([3, 2])
        with text_column:
            st.markdown(
                "- **Normal :** 10 192 images\n"
                "- **Lung Opacity :** 6 012 images\n"
                "- **COVID :** 3 616 images\n"
                "- **Viral Pneumonia :** 1 345 images"
            )
            st.markdown(
                "- **Risque majeur :** Biais des modèles en faveur des classes majoritaires (Normal/Opacités)\n"
                "- **Solutions :** Stratégies d'augmentation de données, pondération des classes et arrêt anticipé\n"
                "- **Métrique clé :** Suivi rigoureux du **F1-score macro** pour une évaluation juste de chaque classe"
            )
        with chart_column:
            show_figure("3.2-distribution_data_class.png", caption="Distribution des classes")

        st.subheader("Analyses statistiques des zones segmentées")
        st.markdown(
            "- **Analyses statistiques menées :** Quantité de pixels, moyenne/écart-type des intensités, ratio de pixels noirs\n"
            "- **Observations :** Intensités moyennes plus faibles pour les cas sains, plus élevées pour les pathologies\n"
            "- **Limites majeures :** Fort chevauchement des distributions statistiques globales entre les classes\n"
            "- **Verdict :** Impossible de classifier par de simples seuils statistiques $\rightarrow$ recours obligatoire au Machine/Deep Learning"
        )
        show_figure(
            "3.4-distribution_pixel_mean.png",
            caption="Distributions et boxplots des moyennes d'intensité par image segmentée",
        )

        st.markdown("**Valeurs extrêmes de moyenne d'intensité**")
        dark_column, bright_column = st.columns(2)
        with dark_column:
            show_figure("3.5-smallest_mean.png", caption="Images les plus sombres")
        with bright_column:
            show_figure("3.5-largest_px_mean.png", caption="Images les plus claires")
        st.markdown(
            "- **Biais d'acquisition :** L'intensité globale traduit le contraste, l'appareil et l'opacité technique de l'image\n"
            "- **Contradiction visuelle :** Des cas normaux apparaissent parmi les plus clairs/sombres et inversement\n"
            "- **Bilan :** Nécessité d'homogénéiser les contrastes (CLAHE) pour masquer ces variations techniques parasites"
        )
    elif section.startswith("3."):
        section_header(
            3,
            "Pré-traitement des données",
            "Homogénéiser les images et concentrer l'apprentissage sur les poumons",
        )
        st.markdown(
            "- **Objectif :** Réduire le bruit technique, homogénéiser les intensités et supprimer les artefacts d'acquisition\n"
            "- **Mode :** Pipeline appliqué hors ligne (offline) avec stockage des images traitées dans `data/processed`"
        )

        st.subheader("Les étapes du pipeline")
        st.markdown(
            "1. **Image** : conversion en niveaux de gris, redimensionnement en 299 × 299 et passage en float32\n"
            "2. **Masque** : binarisation après redimensionnement en 299 × 299 (interpolation adaptée)\n"
            "3. **Masquage** : application du masque pour isoler la région pulmonaire\n"
            "4. **Recadrage** : suppression du padding autour des poumons (maintien du ratio d'aspect)\n"
            "5. **Mise au carré** : re-redimensionnement en 299 × 299\n"
            "6. **CLAHE** : amélioration locale du contraste, en limitant l'amplification du bruit"
        )
        show_figure(
            "4-sample_data_processed.png",
            caption="Échantillon d'images pré-traitées et étapes du processus",
        )

        choice_column, online_column = st.columns(2)
        with choice_column:
            st.subheader("Choix et limites")
            st.markdown(
                "- **Masquage** : focalise les modèles sur les poumons et élimine les corrélations parasites (fond, annotations)\n"
                "- **Limite** : l'hypothèse n'a pas été validée par une étude d'ablation comparant l'image brute vs segmentée à conditions égales\n"
                "- **CLAHE** : indispensable pour faire ressortir les opacités et infiltrats discrets"
            )
        with online_column:
            st.subheader("Normalisation et augmentation online")
            st.markdown(
                "- **Data Augmentation (Train uniquement) :** Appliquée dynamiquement en mémoire (rotations, translations, contraste)\n"
                "- **Pas d'écriture disque :** Évite de surcharger le stockage local et d'induire du surapprentissage\n"
                "- **Bénéfice :** Renforce massivement la robustesse et la généralisation du modèle profond"
            )
    elif section.startswith("4."):
        section_header(
            4,
            "Extraction des caractéristiques",
            "Représenter chaque image par un vecteur numérique pour le machine learning classique",
        )
        st.markdown(
            "- **Contrainte ML classique :** Les modèles classiques requièrent des vecteurs de caractéristiques structurés\n"
            "- **Isolation du signal :** Extraction de caractéristiques **uniquement sur les pixels non nuls** (région des poumons)\n"
            "- **Bénéfice :** Élimine complètement les corrélations parasites et l'influence du fond de l'image"
        )

        stat_tab, lbp_tab, hog_tab = st.tabs(["STAT", "LBP", "HOG"])
        with stat_tab:
            st.markdown(
                "**Caractéristiques statistiques (15 dimensions)** : décrivent la distribution des niveaux de gris pulmonaires."
            )
            st.markdown(
                "- **Tendance centrale :** Mean, Median, Min, Max\n"
                "- **Dispersion et forme :** Std, Skewness (asymétrie), Kurtosis (concentration)\n"
                "- **Répartition fine :** Percentiles (P5, P25, P75, P95)\n"
                "- **Texture et contours :** Shannon Entropy, variance du Laplacien (netteté), statistiques du gradient"
            )
        with lbp_tab:
            st.markdown(
                "**Local Binary Patterns (LBP)** : descripteurs de texture comparant chaque pixel à ses voisins pour produire un motif binaire, résumé sous forme d'histogramme. La variante uniforme est utilisée pour réduire la dimension tout en conservant les motifs structurés."
            )
            show_figure("5.1-lbp_features.png", caption="Extraction des caractéristiques LBP")
        with hog_tab:
            st.markdown(
                "**Histogram of Oriented Gradients (HOG)** : capturent les formes et les contours via la distribution des orientations de gradients par cellule, normalisée localement. Complémentaires des STAT, ils décrivent l'organisation spatiale des structures."
            )
            show_figure("5.1-hog_features.png", caption="Extraction des caractéristiques HOG")

        st.subheader("Combinaisons de caractéristiques testées")
        st.markdown(
            "- **Complémentarité :** STAT (distribution globale des tons), LBP (textures locales), et HOG (contours et formes géométriques)\n"
            "- **Méthodologie :** Entraînement et validation croisée sur chaque famille de descripteurs et leurs combinaisons"
        )
        _, table_column, _ = st.columns([1, 2, 1])
        with table_column:
            show_figure(
                "5.1-features_combinations.png",
                caption="Combinaisons de caractéristiques et nombre associé",
            )
    elif section.startswith("5."):
        section_header(
            5,
            "Approche baseline",
            "SVM comme modèle de référence sur les caractéristiques extraites",
        )
        st.markdown(
            "- **Algorithme de référence :** Support Vector Machine (SVM) maximisant la marge entre les frontières de décision\n"
            "- **Haute dimension :** Particulièrement performant sur de grands vecteurs de caractéristiques (ex. STAT + LBP + HOG)\n"
            "- **Gestion du déséquilibre :** Utilisation du paramètre `class_weight=\"balanced\"` pour équilibrer l'importance des classes"
        )

        impl_column, metric_column = st.columns(2)
        with impl_column:
            st.subheader("Implémentation et standardisation")
            st.markdown(
                "- **Standardisation requise :** Le SVM est sensible à l'échelle des variables ; utilisation de `StandardScaler` indispensable\n"
                "- **Mode d'évaluation :** Entraînement séparé sur chaque famille de caractéristiques et combinaisons complexes"
            )
        with metric_column:
            st.subheader("Métriques observées")
            st.markdown(
                "- **F1-score macro** : métrique principale, adaptée au multiclasse déséquilibré\n"
                "- **Rappel COVID** : proportion de cas COVID détectés (un faux négatif est critique)\n"
                "- **Précision COVID** : pour vérifier qu'un rappel élevé ne s'accompagne pas de trop de faux positifs"
            )

        st.subheader("Résultats baseline")
        st.markdown(
            "- **Performances individuelles :** HOG (76,7 % F1 macro) > STAT (68,0 %) > LBP (67,0 %)\n"
            "- **Synergie des descripteurs :** Les combinaisons améliorent systématiquement les scores\n"
            "- **Champion Baseline :** **STAT + LBP + HOG** obtient un **F1-score macro de 82,8 %**\n"
            "- **Hétérogénéité des classes :**\n"
            "  - **Viral Pneumonia :** Très bien détectée (F1 0,90 ; rappel 94 %) malgré un petit effectif\n"
            "  - **Normal & Lung Opacity :** Correctement distinguées (F1 0,87 & 0,82)\n"
            "  - **COVID :** La plus complexe (F1 0,73 ; précision 70 % ; rappel 75 %)"
        )
        results_column, report_column = st.columns(2)
        with results_column:
            show_figure("5.2.1-svm_features_results.png", caption="Synthèse des résultats SVM")
        with report_column:
            show_figure(
                "5.2.1-svm_classification_report.png",
                caption="Rapport de classification SVM",
            )

        st.subheader("Optimisation")
        st.markdown(
            "- **Pipeline optimisé :** StandardScaler $\rightarrow$ Réduction de dimension PCA (95 % variance) $\rightarrow$ SVM (SVC)\n"
            "- **Méthode :** Recherche d'hyperparamètres C et gamma via `RandomizedSearchCV` (5-fold cross-validation)\n"
            "- **Bénéfices PCA :** Réduction de la redondance des 617 variables et diminution massive du temps de calcul\n"
            "- **Résultat :** Hausse du **F1-score macro à 84,0 %** (gain de +1,2 point)\n"
            "- **Comportement COVID :** Précision en hausse (77 % vs 70 %), mais rappel en léger retrait (72 % vs 75 %)"
        )
        optim_report_column, optim_matrix_column = st.columns(2)
        with optim_report_column:
            show_figure(
                "5.3.1-optim_svm_classification_report.png",
                caption="Rapport de classification — SVM optimisé",
            )
        with optim_matrix_column:
            show_figure(
                "5.3.1-optim_svm_confusion_matrix.png",
                caption="Matrice de confusion — SVM optimisé",
            )

        st.subheader("Conclusion")
        st.markdown(
            "- **Baseline solide :** F1-score macro de 84 % atteint après optimisation\n"
            "- **Palier de performance :** Les gains marginaux indiquent que le classifieur n'est pas le facteur limitant\n"
            "- **Verdict :** Le pouvoir discriminant des caractéristiques manuelles (STAT, LBP, HOG) atteint ses limites physiques"
        )
        st.info(
            "Le **KNN** a également été testé comme baseline. Sa meilleure configuration (STAT + LBP) atteint un F1 macro de 74,5 % (75,5 % après optimisation), nettement en dessous du SVM. Le KNN souffre de la grande dimension de l'espace des caractéristiques (curse of dimensionality)."
        )
    elif section.startswith("6."):
        section_header(
            6,
            "Approche ensemble",
            "XGBoost sur les caractéristiques extraites",
        )
        st.markdown(
            "- **Algorithme d'ensemble :** Boosting de gradients sur des arbres de décision séquentiels (XGBoost)\n"
            "- **Avantages clés :** Extrêmement efficace sur données tabulaires, capture native des relations non linéaires"
        )

        impl_column, metric_column = st.columns(2)
        with impl_column:
            st.subheader("Implémentation")
            st.markdown(
                "- **Sans standardisation :** Les arbres de décision séparent par seuil et sont insensibles aux échelles de variables\n"
                "- **Évaluation :** Entraînement initial sur les configurations brutes par défaut avant réglage fin"
            )
        with metric_column:
            st.subheader("Métriques observées")
            st.markdown(
                "- **F1-score macro** : métrique principale, adaptée au multiclasse déséquilibré\n"
                "- **Rappel COVID** : proportion de cas COVID détectés (faux négatifs critiques)\n"
                "- **Précision COVID** : pour équilibrer rappel élevé et faux positifs"
            )

        st.subheader("Résultats")
        st.markdown(
            "- **Domination baseline :** XGBoost surpasse systématiquement le SVM et le KNN sur tous les jeux de données\n"
            "- **Meilleure performance brute :** **STAT + LBP + HOG** atteint un **F1-score macro de 86,2 %** (sans aucune optimisation)\n"
            "- **Synthèse par classe :**\n"
            "  - **Normal & Viral Pneumonia :** Excellents scores (F1 0,90 et 0,92)\n"
            "  - **Lung Opacity :** Très stable (F1 0,84)\n"
            "  - **COVID :** Reste la classe la plus complexe (F1 0,79 ; précision 86 % ; rappel 73 %)"
        )
        results_column, report_column = st.columns(2)
        with results_column:
            show_figure("6.1.1-xgb_features_results.png", caption="Synthèse des résultats XGBoost")
        with report_column:
            show_figure(
                "6.1.1-xgb_classification_report.png",
                caption="Rapport de classification XGBoost",
            )

        st.subheader("Optimisation")
        st.markdown(
            "- **Stratégie d'optimisation :** Exploration large suivie d'un affinage ciblé via `RandomizedSearchCV` (5 folds)\n"
            "- **Variables ajustées :** `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `min_child_weight` et `gamma`\n"
            "- **Performances finales :** Progression du **F1-score macro à 87,0 %** (+0,8 %)\n"
            "- **Comportement COVID :** Amélioration d'un point (F1 0,80, précision 87 %, rappel 74 %)"
        )
        optim_report_column, optim_matrix_column = st.columns(2)
        with optim_report_column:
            show_figure(
                "6.2.1-optim_xgb_classification_report.png",
                caption="Rapport de classification — XGBoost optimisé",
            )
        with optim_matrix_column:
            show_figure(
                "6.2.1-optim_xgb_confusion_matrix.png",
                caption="Matrice de confusion — XGBoost optimisé",
            )

        st.subheader("Conclusion")
        st.markdown(
            "- **Champion classique :** XGBoost est le meilleur modèle sur descripteurs manuels avec **87,0 % de F1 macro**\n"
            "- **Saturation du signal :** Les gains marginaux d'optimisation confirment la saturation des caractéristiques manuelles\n"
            "- **Transition logique :** Nécessité de basculer vers le **Deep Learning** pour que le modèle extrait lui-même ses propres descripteurs"
        )
        st.info(
            "**Limite** : contrairement au SVM, ce XGBoost n'intègre pas de compensation du déséquilibre des classes. Des pistes comme la pondération des observations (`sample_weight`) ou le sur-échantillonnage (SMOTE) pourraient améliorer la détection des classes minoritaires."
        )
    elif section.startswith("7."):
        section_header(
            7,
            "Approche CNN – ResNet-50",
            "Transfer learning à partir d'un réseau pré-entraîné sur ImageNet",
        )
        st.markdown(
            "- **Architecture de référence :** ResNet-50 (50 couches de convolutions apprenables)\n"
            "- **Skip Connections :** Connexions résiduelles limitant la disparition du gradient et stabilisant l'optimisation\n"
            "- **Stratégie de Transfer Learning :**\n"
            "  - Initialisation des poids pré-entraînés sur **ImageNet**\n"
            "  - Remplacement de la tête de classification par une couche linéaire à **4 classes**\n"
            "  - Fine-tuning complet de tous les paramètres sur nos radiographies segmentées + CLAHE"
        )
        show_figure("7.1.2-resnet_architecture.png", caption="Architecture ResNet et connexions résiduelles")

        st.subheader("Implémentation et entraînement")
        impl_column, params_column = st.columns(2)
        with impl_column:
            st.markdown(
                "- Entrée **224 × 224 × 3**, normalisation **ImageNet**\n"
                "- **Data augmentation** (entraînement uniquement) : flip horizontal, "
                "transformation affine, variations luminosité/contraste, perspective, "
                "Random Erasing\n"
                "- **PyTorch**, perte **Cross Entropy**, optimiseur **AdamW** (lr 5e-5)\n"
                "- **ReduceLROnPlateau** (facteur 0,5 ; patience 2) sur le F1 macro validation\n"
                "- **Early Stopping** (patience 5), meilleur modèle conservé"
            )
        with params_column:
            show_figure("7.1.2-resnet_parametres.png", caption="Paramètres d'entraînement")

        curve_loss_column, curve_f1_column = st.columns(2)
        with curve_loss_column:
            show_figure("7.1.2-resnet_loss_learning_curve.png", caption="Courbe de perte")
        with curve_f1_column:
            show_figure("7.1.2-resnet_f1macro_learning_curve.png", caption="Courbe de F1 macro")
        st.markdown(
            "L'Early Stopping interrompt l'entraînement après **23 époques**. Le F1 de validation "
            "plafonne autour de 94–95 % tandis que celui d'entraînement approche 99 % : un "
            "surapprentissage **modéré**, bien maîtrisé par la stratégie retenue."
        )

        st.subheader("Résultats")
        st.markdown(
            "- **Saut de performance massif :** **F1-score macro de 93,8 %** et **Accuracy globale de 93,0 %**\n"
            "- **Robustesse au déséquilibre :** Écart minime entre F1 macro et F1 pondéré\n"
            "- **Analyse par classe :**\n"
            "  - **Viral Pneumonia :** Détection exceptionnelle (F1 0,96 ; précision 98 %)\n"
            "  - **Normal & COVID :** Très performantes (F1 0,94 et 0,93 ; précision COVID de 96 %)\n"
            "  - **Lung Opacity :** Reste la classe la plus complexe (F1 0,91)\n"
            "  - **Point d'attention métier :** Rappel COVID de 91 % ($\approx$ 9 % de faux négatifs à surveiller en clinique)"
        )
        report_column, matrix_column = st.columns(2)
        with report_column:
            show_figure("7.2.1-resnet50_classification_report.png", caption="Rapport de classification ResNet-50")
        with matrix_column:
            show_figure("7.2.1-resnet50_confusion_matrix.png", caption="Matrice de confusion ResNet-50")
    elif section.startswith("8."):
        section_header(
            8,
            "Custom CNN from scratch",
            "Un réseau convolutif conçu et entraîné entièrement depuis zéro",
        )
        st.markdown(
            "- **Entraînement \"from scratch\" :** Zéro pré-entraînement pour isoler l'apport réel du Transfer Learning\n"
            "- **Architecture inspirée de VGG :** 5 blocs convolutifs doublant le nombre de filtres (32 $\rightarrow$ 512) avec MaxPooling\n"
            "- **Régularisation et réduction :**\n"
            "  - **Global Average Pooling (GAP) :** Réduction massive du nombre de paramètres pour limiter le surapprentissage\n"
            "  - **Tête Fully Connected :** Deux couches denses avec des taux de **Dropout** régulés (0,4 et 0,3)"
        )
        show_figure("7.1.3-cnn_fscratch_architecture.png", caption="Architecture du Custom CNN inspiré de VGG")

        st.subheader("Implémentation et entraînement")
        st.markdown(
            "- Entrée **224 × 224**, **pas de normalisation ImageNet** (aucun poids pré-entraîné)\n"
            "- **Mêmes techniques de data augmentation** que ResNet-50 (d'autant plus utiles "
            "sans pré-entraînement)\n"
            "- **PyTorch**, perte **Cross Entropy**, optimiseur **Adam** (lr 1e-3, plus faible "
            "adapté à l'entraînement from scratch)\n"
            "- **ReduceLROnPlateau** (patience 5) et **Early Stopping** (patience 10)\n"
            "- **Gradient clipping** (norme max 1,0), déterminant pour stabiliser l'entraînement"
        )
        show_figure("7.1.3-cnn_fscratch_learning_curves.png", caption="Courbes d'apprentissage — perte et F1 macro")
        st.markdown(
            "Après des oscillations initiales, la réduction du taux d'apprentissage stabilise "
            "l'entraînement vers l'époque 10 ; l'Early Stopping arrête à **28 époques**."
        )

        st.subheader("Résultats")
        st.markdown(
            "- **Performances très solides :** **91,0 % de F1-score macro** et **Accuracy globale de 90,6 %**\n"
            "- **Verdict :** Seulement 3 points de retard sur ResNet-50, un score impressionnant sans pré-entraînement !\n"
            "- **Profil des classes :**\n"
            "  - **Viral Pneumonia :** Excellente performance (F1 0,95)\n"
            "  - **Normal :** Stable (F1 0,93)\n"
            "  - **COVID :** Très convenable (F1 0,88 ; précision 90 % ; rappel 85 %)\n"
            "  - **Lung Opacity :** Reste en retrait (F1 0,88)"
        )
        cnn_report_column, cnn_matrix_column = st.columns(2)
        with cnn_report_column:
            show_figure("7.2.2-cnn_fscratch_classification_report.png", caption="Rapport de classification Custom CNN")
        with cnn_matrix_column:
            show_figure("7.2.2-cnn_fscratch_confusion_matrix.png", caption="Matrice de confusion Custom CNN")

        st.subheader("Conclusion — Deep Learning")
        st.markdown(
            "- **Saut qualitatif du Deep Learning :** ResNet-50 (93,8 % F1) et Custom CNN (91,0 % F1) écrasent le machine learning classique\n"
            "- **Apport du Transfer Learning :** Le gain de +2,8 points valide l'efficacité des poids d'initialisation pré-entraînés (ImageNet)\n"
            "- **Performance brute \"from scratch\" :** Le Custom CNN démontre qu'avec une forte régularisation (augmentation, dropout, gradient clipping), entraîner un modèle à partir de zéro reste extrêmement compétitif\n"
            "- **Constat d'ambiguïté visuelle :** Les deux architectures butent sur les mêmes confusions (Normal ↔ Opacités, COVID ↔ Normal) ↔ traduit une **ambiguïté clinique intrinsèque** des images plutôt qu'un défaut de modélisation"
        )
    elif section.startswith("9."):
        section_header(
            9,
            "Interprétabilité",
            "Comprendre les décisions du modèle avec Grad-CAM",
        )
        st.markdown(
            "- **Casser l'effet \"boîte noire\" :** Grad-CAM rend explicites et visuelles les décisions prises par le modèle profond\n"
            "- **Mécanisme :** Exploitation des gradients rétropropagés pour pondérer les activations et générer une **heatmap** d'influence\n"
            "- **Suivi multi-niveaux :** Génération des heatmaps à la sortie des **4 blocs convolutifs** de ResNet-50 pour observer l'abstraction :\n"
            "  - **conv_block_1 (bas niveau) :** Détection des textures, contours et formes générales de la cage thoracique\n"
            "  - **conv_block_4 (haut niveau) :** Focus précis et abstrait sur les anomalies pathologiques cibles"
        )

        st.subheader("Exemple 1 — Prédiction correcte")
        show_figure("7.3-resnet50_gradcam_1.png", caption="Grad-CAM ResNet-50 — prédiction correcte")
        st.markdown(
            "L'évolution est conforme au comportement attendu : conv_block_1 active largement la "
            "région pulmonaire et ses contours, les blocs intermédiaires localisent des motifs, "
            "et le dernier bloc se **focalise** sur les zones ayant conduit à la classification."
        )

        st.subheader("Exemple 2 — Erreur avec forte confiance")
        st.markdown("Radiographie **COVID** classée à tort **Normal** avec une probabilité de 99,99 %.")
        show_figure("7.3-resnet50_gradcam_3.png", caption="Grad-CAM ResNet-50 — erreur à forte confiance")
        st.markdown(
            "Le modèle se concentre pourtant sur des régions **anatomiquement cohérentes** "
            "(poumon gauche au dernier bloc). L'erreur ne vient donc pas d'une mauvaise "
            "localisation mais d'une **interprétation incorrecte** d'une zone pertinente : une "
            "localisation correcte ne garantit pas une interprétation correcte."
        )

        st.subheader("Exemple 3 — Erreur avec faible confiance")
        st.markdown("Radiographie **Normal** classée **Lung Opacity** avec une probabilité de 39,39 %.")
        show_figure("7.3-resnet50_gradcam_2.png", caption="Grad-CAM ResNet-50 — erreur à faible confiance")
        st.markdown(
            "Les activations se concentrent progressivement sur une région du poumon droit, à la "
            "texture légèrement plus dense. La **faible probabilité** traduit une hésitation du "
            "modèle face à une **ambiguïté visuelle**, plutôt qu'une mauvaise localisation."
        )

        st.subheader("Observations")
        st.markdown(
            "- Le modèle fonde globalement ses décisions sur des **régions pertinentes** des "
            "poumons\n"
            "- Les représentations évoluent du **local** (premières couches) vers le "
            "**discriminant** (dernières couches)\n"
            "- Les erreurs proviennent souvent d'une **mauvaise interprétation** de "
            "caractéristiques bien localisées, non d'une attention mal placée\n"
            "- Cela renforce la confiance dans le comportement global du modèle, tout en "
            "révélant les limites face à des pathologies aux manifestations proches"
        )
    elif section.startswith("10."):
        section_header(
            10,
            "Conclusion et résultats finaux",
            "Comparaison des approches, limites et perspectives",
        )
        st.markdown(
            "- **Pipeline de bout en bout :** Exploration de données, prétraitement, baselines classiques et architectures profondes\n"
            "- **Évaluation unifiée :** Tous les modèles ont été comparés sur le **même jeu de test figé** pour une rigueur scientifique totale\n"
            "- **Double priorité :** Optimiser le **F1-score macro** pour contrer le déséquilibre, tout en surveillant le **Rappel COVID** (réduire les faux négatifs cliniques)"
        )
        show_figure("8-synthesis_results.png", caption="Tableau récapitulatif des résultats par modèle")

        st.subheader("Résultats")
        st.markdown(
            "- **Victoire incontestable du Deep Learning :** Surpasse largement le machine learning sur toutes les métriques\n"
            "- **ResNet-50 vainqueur de l'étude :** **F1-score macro de 93,8 %**, **Rappel COVID de 91,0 %**, et **Précision COVID de 96,0 %**\n"
            "- **Explication scientifique :** Les CNN encodent de manière spatiale les textures et opacités complexes, là où les descripteurs manuels perdent l'information de structure"
        )

        limits_column, perspectives_column = st.columns(2)
        with limits_column:
            st.subheader("Difficultés et limites")
            st.markdown(
                "- **Déséquilibre des classes**, non traité spécifiquement pour XGBoost\n"
                "- **Coût de calcul** des CNN (régularisation + arrêt anticipé nécessaires)\n"
                "- Pas de **courbes d'apprentissage** pour SVM/KNN/XGBoost (biais/variance)\n"
                "- Pas de **courbes Precision-Recall** par classe (compromis selon le seuil)\n"
                "- **Découpage** stratifié mais non garanti indépendant par patient/source (risque de fuite de données)\n"
                "- **Masquage** non évalué par une étude d'ablation"
            )
        with perspectives_column:
            st.subheader("Perspectives")
            st.markdown(
                "- Gérer le déséquilibre dans XGBoost (`sample_weight`, SMOTE)\n"
                "- Comparer d'autres ensembles (Random Forest) et architectures pré-entraînées (DenseNet, EfficientNet)\n"
                "- **Étude d'ablation** du masquage avec ResNet-50\n"
                "- Ajuster le **seuil COVID** via les courbes Precision-Recall\n"
                "- Découpage par groupes + **jeu de données externe**\n"
                "- Aller vers la **segmentation** des zones pathologiques"
            )

        st.success(
            "ResNet-50 constitue le meilleur modèle du projet. Grad-CAM montre l'intérêt de ne "
            "pas limiter l'évaluation d'un modèle médical à ses seules performances "
            "quantitatives, avant d'envisager un usage applicatif réel."
        )
    else:
        demo.render()


if __name__ == "__main__":
    main()
