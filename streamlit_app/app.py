import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Streamlit runs this file directly, so the repository root may not be importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.components import metric_row, section_header, show_figure
from streamlit_app.sections import demo


st.set_page_config(page_title="Analyse de radiographies COVID-19", page_icon="🫁", layout="wide")


def scroll_to_top_on_change(section: str) -> None:
    if st.session_state.get("_last_section") != section:
        st.session_state._last_section = section
        # Section label embedded so the iframe HTML differs and the script re-runs each change.
        components.html(
            f"""
            <script>
            const marker = {section!r};
            const doc = window.parent.document;
            const selectors = [
                'section.main',
                '[data-testid="stMain"]',
                '[data-testid="stAppViewContainer"]',
                '.main',
            ];
            function scrollTop() {{
                for (const selector of selectors) {{
                    const element = doc.querySelector(selector);
                    if (element) {{ element.scrollTo({{top: 0, behavior: 'instant'}}); }}
                }}
                window.parent.scrollTo(0, 0);
            }}
            scrollTop();
            setTimeout(scrollTop, 50);
            setTimeout(scrollTop, 200);
            </script>
            """,
            height=0,
        )


def render_static_section(number: int, title: str, subtitle: str, figures: list[str]) -> None:
    section_header(number, title, subtitle)
    for figure in figures:
        show_figure(figure)


def main() -> None:
    st.sidebar.title("Soutenance")
    section = st.sidebar.radio(
        "Navigation",
        [
            "Page de garde",
            "1. Introduction",
            "2. Exploration et analyse des données",
            "3. Pré-traitement des données",
            "4. Extraction des caractéristiques",
            "5. Approche baseline",
            "6. Approche ensemble",
            "7. Approche CNN – ResNet-50",
            "8. Custom CNN from scratch",
            "9. Interprétabilité",
            "10. Conclusion et résultats finaux",
            "11. Démonstration ResNet-50",
        ],
    )

    scroll_to_top_on_change(section)

    if section == "Page de garde":
        _, center_column, _ = st.columns([1, 4, 1])
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
    elif section.startswith("1."):
        section_header(
            1,
            "Introduction",
            "Intelligence artificielle et analyse de radiographies pulmonaires",
        )
        st.subheader("Contexte")
        st.markdown(
            "La pandémie de Covid-19 a souligné le besoin d'outils de diagnostic "
            "rapides, fiables et accessibles. La PCR reste la méthode de référence, "
            "mais sa disponibilité, son coût ou son délai d'analyse peuvent limiter "
            "son utilisation dans certains contextes. Les radiographies pulmonaires "
            "constituent une source d'information complémentaire pour détecter les "
            "atteintes respiratoires."
        )
        st.markdown(
            "Dans ce contexte, la computer vision et le deep learning permettent "
            "d'extraire automatiquement des caractéristiques complexes afin de "
            "distinguer les cas Covid-19, les cas sains et les autres pneumonies."
        )

        scope_column, objective_column = st.columns(2)
        with scope_column:
            st.subheader("Périmètre")
            st.markdown(
                "- Dataset public COVID-19 Radiography Database\n"
                "- Radiographies pulmonaires 2D en niveaux de gris\n"
                "- Quatre classes : Normal, COVID, Lung Opacity et Viral Pneumonia\n"
                "- Masques pulmonaires pour isoler la région d'intérêt\n"
                "- Comparaison entre machine learning classique et deep learning"
            )
        with objective_column:
            st.subheader("Objectif")
            st.markdown(
                "Développer un système capable de classifier automatiquement les "
                "radiographies pulmonaires et comparer plusieurs approches afin "
                "d'identifier les architectures les plus performantes."
            )
            st.markdown(
                "L'étude évalue aussi l'impact de la segmentation sur la robustesse "
                "et la précision, puis utilise Grad-CAM pour visualiser les régions "
                "ayant influencé la décision du modèle."
            )

        st.subheader("Déroulé de la soutenance")
        st.markdown(
            "- **Exploration et analyse des données** : structure du dataset, classes, "
            "dimensions et premières observations\n"
            "- **Pré-traitement** : segmentation pulmonaire, homogénéisation et amélioration "
            "du contraste\n"
            "- **Machine Learning** : baseline avec SVM et KNN, puis XGBoost\n"
            "- **Deep learning** : comparaison entre un Custom CNN et ResNet-50\n"
            "- **Interprétabilité et bilan** : Grad-CAM, comparaison finale des résultats et "
            "démonstration interactive avec ResNet-50"
        )

        # st.caption(
        #     "Le projet évalue le potentiel de l'IA comme outil d'aide au diagnostic ; "
        #     "il ne remplace pas l'expertise médicale."
        # )
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
                # ("Masques", "1 par image"),
            ]
        )
        st.markdown(
            "Le jeu de données regroupe quatre pathologies pulmonaires. Chaque radiographie "
            "est en niveaux de gris et dispose d'un masque de segmentation associé, qui isole "
            "la région des poumons."
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
                "- Normal : 10 192 images\n"
                "- Lung Opacity : 6 012 images\n"
                "- COVID : 3 616 images\n"
                "- Viral Pneumonia : 1 345 images"
            )
            st.markdown(
                "Ce déséquilibre pousse les modèles à privilégier les classes majoritaires. "
                "Il impose des stratégies adaptées (augmentation, poids de classes) et le suivi "
                "du **F1-score macro** pour évaluer équitablement toutes les catégories."
            )
        with chart_column:
            show_figure("3.2-distribution_data_class.png", caption="Distribution des classes")

        st.subheader("Analyses statistiques des zones segmentées")
        st.markdown(
            "Plusieurs analyses ont été menées sur les régions pulmonaires segmentées : "
            "quantité de pixels, moyenne et écart-type des intensités, ratio de pixels noirs. "
            "Exemple ci-dessous avec la **moyenne d'intensité des pixels** par image segmentée."
        )
        show_figure(
            "3.4-distribution_pixel_mean.png",
            caption="Distributions et boxplots des moyennes d'intensité par image segmentée",
        )
        st.markdown(
            "La classe Normal tend vers des intensités plus faibles et les classes pathologiques "
            "vers des valeurs plus élevées, mais les distributions se **chevauchent fortement**. "
            "Aucune statistique globale ne suffit, à elle seule, à distinguer clairement les "
            "pathologies : cela justifie le recours à des modèles capables d'apprendre des "
            "caractéristiques plus complexes."
        )

        st.markdown("**Valeurs extrêmes de moyenne d'intensité**")
        dark_column, bright_column = st.columns(2)
        with dark_column:
            show_figure("3.5-smallest_mean.png", caption="Images les plus sombres")
        with bright_column:
            show_figure("3.5-largest_px_mean.png", caption="Images les plus claires")
        st.markdown(
            "Les cas extrêmes confirment que l'intensité reflète surtout des différences "
            "d'opacité, de contraste et de conditions d'acquisition, et non directement la "
            "pathologie : des images Normal apparaissent parmi les plus claires et inversement."
        )
    elif section.startswith("3."):
        section_header(
            3,
            "Pré-traitement des données",
            "Homogénéiser les images et concentrer l'apprentissage sur les poumons",
        )
        st.markdown(
            "Le pré-traitement homogénéise les radiographies et réduit les variations liées "
            "aux conditions d'acquisition. La chaîne est appliquée hors ligne, puis les images "
            "sont sauvegardées pour la modélisation."
        )

        st.subheader("Les étapes du pipeline")
        st.markdown(
            "1. **Image** : conversion en niveaux de gris, redimensionnement en 299 × 299 et "
            "passage en float32\n"
            "2. **Masque** : conversion en niveaux de gris, redimensionnement en 299 × 299 "
            "(interpolation adaptée) puis binarisation\n"
            "3. **Masquage** : application du masque pour isoler la région pulmonaire\n"
            "4. **Recadrage** : suppression du padding autour des poumons, en conservant les "
            "proportions pour ne pas distordre l'information\n"
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
                "- **Masquage** : hypothèse que l'information utile est surtout dans les poumons ; "
                "réduit les corrélations parasites (fond, annotations, artefacts)\n"
                "- **Limite** : cette hypothèse n'a pas été validée par une étude d'ablation "
                "(comparaison image complète vs image masquée à conditions égales)\n"
                "- **CLAHE** : adapté aux radiographies où certaines anomalies ont de faibles "
                "variations d'intensité"
            )
        with online_column:
            st.subheader("Normalisation et augmentation online")
            st.markdown(
                "La normalisation des intensités et la data augmentation ne sont **pas "
                "appliquées hors ligne**, mais dynamiquement à l'entraînement, à chaque époque : "
                "rotations légères, translations, variations de luminosité et de contraste. "
                "Cela augmente la diversité des données sans créer d'images sur disque et "
                "réduit le risque de surapprentissage."
            )
    elif section.startswith("4."):
        section_header(
            4,
            "Extraction des caractéristiques",
            "Représenter chaque image par un vecteur numérique pour le machine learning classique",
        )
        st.markdown(
            "Les algorithmes de machine learning classiques n'exploitent pas directement les "
            "images. Chaque radiographie est donc décrite par un vecteur de caractéristiques, "
            "extraites **uniquement à partir des pixels non nuls** des images segmentées pour "
            "éviter l'influence du fond."
        )

        stat_tab, lbp_tab, hog_tab = st.tabs(["STAT", "LBP", "HOG"])
        with stat_tab:
            st.markdown(
                "**Caractéristiques statistiques** : décrivent la distribution des niveaux de "
                "gris de la région pulmonaire segmentée."
            )
            st.markdown(
                "- **Mean** : intensité moyenne des pixels\n"
                "- **Std** : dispersion des intensités (hétérogénéité)\n"
                "- **Min / Max** : valeurs extrêmes des niveaux de gris\n"
                "- **Median** : valeur centrale, robuste aux valeurs aberrantes\n"
                "- **Percentiles (P5, P25, P75, P95)** : répartition des intensités\n"
                "- **Skewness** : asymétrie de la distribution\n"
                "- **Kurtosis** : aplatissement ou concentration de la distribution\n"
                "- **Entropy** : complexité des intensités\n"
                "- **Variance du Laplacien** : richesse en contours et netteté\n"
                "- **Gradient magnitude mean / std** : moyenne et dispersion des gradients"
            )
        with lbp_tab:
            st.markdown(
                "**Local Binary Patterns** : descripteurs de texture comparant chaque pixel à "
                "ses voisins pour produire un motif binaire, résumé sous forme d'histogramme. "
                "La variante uniforme est utilisée pour réduire la dimension tout en conservant "
                "les motifs les plus représentatifs."
            )
            show_figure("5.1-lbp_features.png", caption="Extraction des caractéristiques LBP")
        with hog_tab:
            st.markdown(
                "**Histogram of Oriented Gradients** : capturent les formes et les contours via "
                "la distribution des orientations de gradients par cellule, normalisée "
                "localement. Complémentaires des STAT, ils décrivent l'organisation spatiale "
                "des structures."
            )
            show_figure("5.1-hog_features.png", caption="Extraction des caractéristiques HOG")

        st.subheader("Combinaisons de caractéristiques testées")
        st.markdown(
            "Les trois familles sont complémentaires : distribution globale (STAT), textures "
            "locales (LBP) et formes (HOG). Les modèles sont entraînés sur chaque jeu et sur "
            "leurs combinaisons pour identifier la représentation la plus discriminante."
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
            "L'étude baseline se concentre sur le **Support Vector Machine (SVM)**, qui "
            "construit une frontière de décision maximisant la marge entre les classes. Il est "
            "bien adapté aux jeux de caractéristiques de dimension élevée comme ceux incluant "
            "HOG. Le paramètre `class_weight=\"balanced\"` compense le déséquilibre des classes."
        )

        impl_column, metric_column = st.columns(2)
        with impl_column:
            st.subheader("Implémentation et standardisation")
            st.markdown(
                "Pipeline scikit-learn : **StandardScaler** puis **SVC**. La standardisation "
                "est indispensable car le SVM est sensible à l'échelle des variables — sans "
                "elle, une variable de grande amplitude dominerait la frontière de décision. "
                "Le modèle est entraîné sur chaque jeu de caractéristiques et leurs combinaisons."
            )
        with metric_column:
            st.subheader("Métriques observées")
            st.markdown(
                "- **F1-score macro** : métrique principale, adaptée au multiclasse déséquilibré\n"
                "- **Rappel COVID** : proportion de cas COVID détectés (un faux négatif est "
                "critique)\n"
                "- **Précision COVID** : pour vérifier qu'un rappel élevé ne s'accompagne pas "
                "de trop de faux positifs"
            )

        st.subheader("Résultats baseline")
        st.markdown(
            "Utilisées seules, les caractéristiques HOG sont les plus discriminantes "
            "(F1 macro 76,7 %) devant STAT (68,0 %) et LBP (67,0 %). Les combinaisons "
            "améliorent systématiquement les performances : la meilleure est "
            "**STAT + LBP + HOG** avec un **F1 macro de 82,8 %**, confirmant la "
            "complémentarité des trois familles."
        )
        results_column, report_column = st.columns(2)
        with results_column:
            show_figure("5.2.1-svm_features_results.png", caption="Synthèse des résultats SVM")
        with report_column:
            show_figure(
                "5.2.1-svm_classification_report.png",
                caption="Rapport de classification SVM",
            )
        st.markdown(
            "Le rapport de classification confirme des performances hétérogènes : COVID reste "
            "la plus difficile (F1 0,73 ; précision 70 % ; rappel 75 %), tandis que Viral "
            "Pneumonia est très bien reconnue (F1 0,90 ; rappel 94 %) malgré son faible "
            "effectif. Normal (F1 0,87) et Lung Opacity (F1 0,82) sont correctement classées."
        )

        st.subheader("Optimisation")
        st.markdown(
            "Sur la configuration STAT + LBP + HOG : pipeline **StandardScaler → PCA (95 % de "
            "variance) → SVC**, avec optimisation de **C** et **gamma** via "
            "**RandomizedSearchCV** en validation croisée stratifiée à 5 folds. La PCA réduit "
            "la redondance des 617 caractéristiques et le coût de calcul."
        )
        st.markdown(
            "Le modèle optimisé atteint un **F1 macro de 84 %** (contre 82,8 %). Le gain profite "
            "surtout à COVID et Normal : la précision COVID passe de 70 % à 77 %, mais au prix "
            "d'un rappel qui recule de 75 % à 72 %."
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
        st.markdown(
            "La matrice de confusion montre une légère baisse des confusions entre classes "
            "pathologiques, notamment entre Lung Opacity et COVID. Certaines images COVID "
            "restent toutefois classées comme Normal, ce qui illustre la difficulté à séparer "
            "ces deux catégories à partir des seules caractéristiques manuelles."
        )

        st.subheader("Conclusion")
        st.markdown(
            "Le SVM constitue une **baseline solide (F1 macro 84 %)**. Les gains de "
            "l'optimisation restent modestes : le facteur limitant n'est plus le classifieur "
            "mais le **pouvoir discriminant des caractéristiques extraites**, en particulier "
            "pour COVID dont les manifestations chevauchent celles de Lung Opacity."
        )
        st.info(
            "Le **KNN** a également été testé comme baseline. Sa meilleure configuration "
            "(STAT + LBP) atteint un F1 macro de 74,5 % (75,5 % après optimisation), nettement "
            "en dessous du SVM. Sur la classe COVID : précision 70 % et rappel 61 % (baseline), "
            "78 % et 59 % après optimisation. Le KNN souffre de la grande dimension de l'espace "
            "des caractéristiques (curse of dimensionality)."
        )
    elif section.startswith("6."):
        section_header(
            6,
            "Approche ensemble",
            "XGBoost sur les caractéristiques extraites",
        )
        st.markdown(
            "L'approche ensemble s'appuie sur **XGBoost** (Extreme Gradient Boosting), qui "
            "construit successivement des arbres de décision pour corriger progressivement les "
            "erreurs des précédents. Il est réputé efficace sur données tabulaires et capable "
            "de modéliser des relations non linéaires entre les caractéristiques."
        )

        impl_column, metric_column = st.columns(2)
        with impl_column:
            st.subheader("Implémentation")
            st.markdown(
                "Contrairement au SVM et au KNN, XGBoost **ne nécessite pas de "
                "standardisation** : les arbres séparent les données par seuils et sont donc "
                "insensibles à l'échelle des variables. Le modèle est d'abord entraîné avec les "
                "paramètres par défaut de `XGBClassifier`, sur les mêmes jeux de "
                "caractéristiques que la baseline."
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
            "XGBoost dépasse les deux baselines quel que soit le jeu de caractéristiques. "
            "HOG seul atteint 76,4 %, STAT + LBP 80,4 %, et la meilleure combinaison "
            "**STAT + LBP + HOG** atteint un **F1 macro de 86,2 %** — au-dessus du SVM optimisé "
            "(84,0 %) et du KNN (75,5 %)."
        )
        results_column, report_column = st.columns(2)
        with results_column:
            show_figure("6.1.1-xgb_features_results.png", caption="Synthèse des résultats XGBoost")
        with report_column:
            show_figure(
                "6.1.1-xgb_classification_report.png",
                caption="Rapport de classification XGBoost",
            )
        st.markdown(
            "Normal (F1 0,90) et Viral Pneumonia (F1 0,92) sont les mieux reconnues, Lung "
            "Opacity reste élevée (0,84). COVID demeure la plus difficile (F1 0,79 ; précision "
            "86 % ; rappel 73 %) : sa détection reste l'axe d'amélioration principal."
        )

        st.subheader("Optimisation")
        st.markdown(
            "Optimisation en deux temps (exploration large puis affinage) via "
            "**RandomizedSearchCV** en validation croisée stratifiée à 5 folds sur le F1 macro. "
            "Hyperparamètres explorés : nombre d'arbres, taux d'apprentissage, profondeur "
            "maximale, proportion de variables par arbre, minimum d'exemples par nœud et gamma."
        )
        st.markdown(
            "Le modèle optimisé atteint un **F1 macro de 87,0 %** (contre 86,2 %). COVID "
            "progresse légèrement (F1 0,80, +1 point en précision et en rappel) ; Normal 0,91 "
            "et Viral Pneumonia 0,93 conservent d'excellents résultats."
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
        st.markdown(
            "Les confusions COVID/Normal et Normal/Lung Opacity diminuent légèrement. Les "
            "erreurs restantes concernent surtout des cas COVID prédits Normal ou Lung Opacity, "
            "cohérent avec la similarité radiologique de ces pathologies."
        )

        st.subheader("Conclusion")
        st.markdown(
            "XGBoost obtient les **meilleures performances des modèles sur caractéristiques "
            "manuelles (F1 macro 87 %)**, sans nouveau prétraitement. Les gains d'optimisation "
            "modestes montrent que la limite vient désormais du **pouvoir discriminant des "
            "caractéristiques**, ce qui justifie le passage au deep learning."
        )
        st.info(
            "**Limite** : contrairement au SVM, ce XGBoost n'intègre pas de compensation du "
            "déséquilibre des classes. Des pistes comme la pondération des observations "
            "(`sample_weight`) ou le sur-échantillonnage (SMOTE) pourraient améliorer la "
            "détection des classes minoritaires, à évaluer sur le F1 macro et le rappel COVID."
        )
    elif section.startswith("7."):
        section_header(
            7,
            "Approche CNN – ResNet-50",
            "Transfer learning à partir d'un réseau pré-entraîné sur ImageNet",
        )
        st.markdown(
            "**ResNet-50** est un réseau convolutif de référence. Ses **connexions résiduelles** "
            "(skip connections) facilitent l'optimisation des réseaux profonds en limitant la "
            "disparition du gradient. Il compte 50 couches apprenables organisées autour d'une "
            "convolution initiale et de 4 blocs résiduels (layer1 à layer4)."
        )
        st.markdown(
            "**Transfer learning** : le réseau est initialisé avec les poids ImageNet, puis sa "
            "dernière couche (1000 classes) est remplacée par une couche linéaire à **4 sorties** "
            "(Normal, COVID, Lung Opacity, Viral Pneumonia). Tous les paramètres sont ensuite "
            "réentraînés sur les radiographies pré-traitées (segmentées + CLAHE)."
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
            "Le modèle atteint une **accuracy de 93 %** et un **F1 macro de 93,8 %**. Le faible "
            "écart avec le F1 pondéré (93,0 %) confirme une bonne robustesse au déséquilibre."
        )
        report_column, matrix_column = st.columns(2)
        with report_column:
            show_figure("7.2.1-resnet50_classification_report.png", caption="Rapport de classification ResNet-50")
        with matrix_column:
            show_figure("7.2.1-resnet50_confusion_matrix.png", caption="Matrice de confusion ResNet-50")
        st.markdown(
            "Viral Pneumonia est la mieux reconnue (F1 0,96 ; précision 98 %), COVID (F1 0,93 ; "
            "précision 96 % ; rappel 91 %) et Normal (F1 0,94) sont élevées, Lung Opacity reste "
            "la plus difficile (F1 0,91). Les erreurs concernent surtout Lung Opacity ↔ Normal "
            "et COVID → Normal ; le rappel COVID de 91 % (≈ 9 % de cas manqués) reste le point "
            "d'attention métier."
        )
    elif section.startswith("8."):
        section_header(
            8,
            "Custom CNN from scratch",
            "Un réseau convolutif conçu et entraîné entièrement depuis zéro",
        )
        st.markdown(
            "Ce modèle est entraîné **sans aucun pré-entraînement**, pour mesurer l'apport réel "
            "du transfer learning et contrôler entièrement l'architecture. Elle s'inspire de "
            "**VGG** : 5 blocs convolutifs dont le nombre de filtres double (32 → 512) avec "
            "MaxPooling, chacun suivi de **BatchNorm** et **ReLU**."
        )
        st.markdown(
            "En sortie, un **Global Average Pooling** réduit fortement le nombre de paramètres "
            "(moins de surapprentissage), suivi de deux couches fully connected entrecoupées de "
            "**Dropout** (0,4 puis 0,3)."
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
            "Le modèle atteint une **accuracy de 90,6 %** et un **F1 macro de 91 %**, soit "
            "environ 3 points sous ResNet-50, mais solide sans pré-entraînement."
        )
        cnn_report_column, cnn_matrix_column = st.columns(2)
        with cnn_report_column:
            show_figure("7.2.2-cnn_fscratch_classification_report.png", caption="Rapport de classification Custom CNN")
        with cnn_matrix_column:
            show_figure("7.2.2-cnn_fscratch_confusion_matrix.png", caption="Matrice de confusion Custom CNN")
        st.markdown(
            "La hiérarchie par classe est proche de ResNet-50 : Viral Pneumonia la mieux "
            "reconnue (F1 0,95), Normal 0,93, COVID 0,88 (précision 90 % ; rappel 85 %), Lung "
            "Opacity la plus difficile (F1 0,88). Les mêmes confusions dominent : "
            "Lung Opacity ↔ Normal et COVID ↔ Normal."
        )

        st.subheader("Conclusion — Deep Learning")
        st.markdown(
            "Le deep learning améliore nettement la classification par rapport au machine "
            "learning classique : **ResNet-50 atteint 93,8 % de F1 macro**, contre **91 % pour "
            "le Custom CNN**. Cet écart d'environ 3 points illustre l'apport concret du "
            "transfer learning, tout en montrant qu'une architecture entraînée depuis zéro "
            "reste compétitive avec une régularisation adaptée (Dropout, augmentation, gradient "
            "clipping)."
        )
        st.markdown(
            "Les deux modèles rencontrent les **mêmes difficultés** (Normal ↔ Lung Opacity et "
            "COVID ↔ Normal), ce qui suggère une **ambiguïté radiologique intrinsèque** entre "
            "ces pathologies plutôt qu'une limite propre à une architecture."
        )
    elif section.startswith("9."):
        section_header(
            9,
            "Interprétabilité",
            "Comprendre les décisions du modèle avec Grad-CAM",
        )
        st.markdown(
            "Les CNN sont souvent des **boîtes noires**. **Grad-CAM** (Gradient-weighted Class "
            "Activation Mapping) exploite les gradients de la rétropropagation pour pondérer les "
            "cartes de caractéristiques d'une couche convolutive et produire une **heatmap** des "
            "régions ayant le plus contribué à la décision — une interprétation directement liée "
            "au processus de classification."
        )
        st.markdown(
            "Les cartes sont générées à la sortie des **4 blocs convolutifs** de ResNet-50 "
            "(conv_block_1 à conv_block_4), pour observer l'évolution des représentations : des "
            "caractéristiques de bas niveau (contours, textures) vers les zones discriminantes "
            "de la classification finale. Les exemples ci-dessous concernent **ResNet-50**."
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
            "Le projet a couvert toute la chaîne de traitement — analyse, prétraitement, "
            "machine learning et deep learning — évaluée sur un **même jeu de test**. Le "
            "**F1-score macro** est la métrique principale, complétée par le **rappel** et la "
            "**précision COVID**, un faux négatif COVID étant particulièrement critique."
        )
        show_figure("8-synthesis_results.png", caption="Tableau récapitulatif des résultats par modèle")

        st.subheader("Résultats")
        st.markdown(
            "Les approches **Deep Learning surpassent** le machine learning, en performance "
            "globale comme sur la détection de COVID. Le Custom CNN (F1 0,91) dépasse déjà SVM "
            "et XGBoost, et **ResNet-50 obtient les meilleurs résultats** de l'étude : "
            "**F1 macro 0,94 ; rappel COVID 0,91 ; précision COVID 0,96**. Cela confirme "
            "l'intérêt des CNN pour exploiter l'information spatiale et l'apport du transfer "
            "learning."
        )

        limits_column, perspectives_column = st.columns(2)
        with limits_column:
            st.subheader("Difficultés et limites")
            st.markdown(
                "- **Déséquilibre des classes**, non traité spécifiquement pour XGBoost\n"
                "- **Coût de calcul** des CNN (régularisation + arrêt anticipé nécessaires)\n"
                "- Pas de **courbes d'apprentissage** pour SVM/KNN/XGBoost (biais/variance)\n"
                "- Pas de **courbes Precision-Recall** par classe (compromis selon le seuil)\n"
                "- **Découpage** stratifié mais non garanti indépendant par patient/source "
                "(risque de fuite de données)\n"
                "- **Masquage** non évalué par une étude d'ablation"
            )
        with perspectives_column:
            st.subheader("Perspectives")
            st.markdown(
                "- Gérer le déséquilibre dans XGBoost (`sample_weight`, SMOTE)\n"
                "- Comparer d'autres ensembles (Random Forest) et architectures pré-entraînées "
                "(DenseNet, EfficientNet)\n"
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
