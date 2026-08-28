from pathlib import Path

import streamlit as st

from .config import DEMO_DIR, FIGURES_DIR


def section_header(number: int, title: str, subtitle: str | None = None) -> None:
    st.title(f"{number}. {title}")
    if subtitle:
        st.caption(subtitle)


def show_figure(filename: str, caption: str | None = None) -> None:
    figure_path = FIGURES_DIR / filename
    if figure_path.exists():
        st.image(str(figure_path), caption=caption, width="stretch")
    else:
        st.warning(f"Figure indisponible : {filename}")


def metric_row(metrics: list[tuple[str, str]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)


def find_demo_images() -> list[Path]:
    if not DEMO_DIR.exists():
        return []
    return sorted(
        path
        for path in DEMO_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and not path.stem.endswith("_mask")
        and not path.stem.endswith("_steps")
    )
