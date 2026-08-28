import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from ..components import section_header
from ..config import CLASS_NAMES, DEMO_DIR, TORCH_AVAILABLE
from ..core.preprocessing import run_pipeline


# (label affiché, dossier, classe canonique du modèle)
CLASS_FOLDERS = [
    ("COVID", "covid", "COVID"),
    ("Normal", "normal", "Normal"),
    ("Lung Opacity", "lung_opacity", "Lung_Opacity"),
    ("Viral Pneumonia", "Viral Pneumonia", "Viral Pneumonia"),
]


def list_class_images(folder: str) -> list[Path]:
    directory = DEMO_DIR / folder
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and not path.stem.endswith("_mask")
    )


def _select_one(active_folder: str) -> None:
    for _, folder, _ in CLASS_FOLDERS:
        if folder != active_folder:
            st.session_state[f"sel_{folder}"] = None
    st.session_state.active_folder = active_folder


def render() -> None:
    section_header(11, "Démonstration ResNet-50",
                   "Sélection, pré-traitement et prédiction")

    columns = st.columns(4)
    for (display_label, folder, _), column in zip(CLASS_FOLDERS, columns):
        with column:
            st.selectbox(
                display_label,
                list_class_images(folder),
                index=None,
                key=f"sel_{folder}",
                format_func=lambda path: path.name,
                placeholder="Choisir…",
                on_change=_select_one,
                args=(folder,),
            )

    active_folder = st.session_state.get("active_folder")
    selected = st.session_state.get(
        f"sel_{active_folder}") if active_folder else None
    if selected is None:
        return

    canonical_class = next(
        cls for _, folder, cls in CLASS_FOLDERS if folder == active_folder)
    image_bytes = selected.read_bytes()
    mask_path = selected.with_name(f"{selected.stem}_mask{selected.suffix}")
    mask_bytes = mask_path.read_bytes() if mask_path.exists() else None
    raw_image = Image.open(selected).convert("L")

    image_key = hashlib.sha256(image_bytes + (mask_bytes or b"")).hexdigest()
    if st.session_state.get("demo_image_key") != image_key:
        for key in ("demo_steps", "demo_prediction", "demo_cam"):
            st.session_state.pop(key, None)
        st.session_state.demo_image_key = image_key

    if mask_bytes is None:
        st.warning(
            "Aucun masque associé : le pipeline sera exécuté sans masquage pulmonaire.")

    _, left, _, right, _ = st.columns([1, 3, 1, 3, 1])
    with left:
        st.image(raw_image, caption="Image originale", width="stretch")
        if st.button("Pré-traiter", type="primary", width="stretch"):
            st.session_state.demo_steps = run_pipeline(image_bytes, mask_bytes)
            st.session_state.demo_prediction = None
            st.session_state.demo_cam = None

    with right:
        if "demo_steps" in st.session_state:
            clahe_image = np.clip(
                st.session_state.demo_steps["clahe"], 0, 255).astype("uint8")
            st.image(clahe_image, caption="Image pré-traitée", width="stretch")
            if not TORCH_AVAILABLE:
                st.warning(
                    "PyTorch n'est pas installé : la prédiction est indisponible.")
            elif st.button("Prédire avec ResNet-50", type="primary", width="stretch"):
                from ..core.model import predict

                with st.spinner("Prédiction en cours…"):
                    st.session_state.demo_prediction = predict(
                        st.session_state.demo_steps["clahe"])
                st.session_state.demo_cam = None

    prediction = st.session_state.get("demo_prediction")
    if prediction:
        st.subheader("Prédiction")
        col_expected, col_predicted, col_confidence, col_chart = st.columns([
                                                                            1, 1, 1, 1])
        col_expected.metric("Label attendu", canonical_class)
        col_predicted.metric("Classe prédite", str(prediction["class_name"]))
        col_confidence.metric(
            "Confiance", f"{float(prediction['confidence']):.1%}")
        with col_chart:
            probabilities = pd.DataFrame(
                {"Probabilité": [float(p)
                                 for p in prediction["probabilities"]]},
                index=CLASS_NAMES,
            )
            st.bar_chart(probabilities)
        if prediction["class_name"] == canonical_class:
            st.success(
                f"Prédiction correcte — label attendu : {canonical_class}")
        else:
            st.error(
                f"Prédiction incorrecte — label attendu : {canonical_class}")
        if st.button("Interpréter avec Grad-CAM", type="primary", width="stretch"):
            from ..core.gradcam import compute_gradcam
            from ..core.model import load_resnet50

            with st.spinner("Calcul Grad-CAM en cours…"):
                st.session_state.demo_cam = compute_gradcam(
                    load_resnet50(), st.session_state.demo_steps["clahe"]
                )

    if st.session_state.get("demo_cam"):
        st.subheader("Interprétabilité Grad-CAM")
        cam_columns = st.columns(4)
        for column, (name, visualization) in zip(cam_columns, st.session_state.demo_cam.items()):
            column.image(visualization, caption=name, width="stretch")
