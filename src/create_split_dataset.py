import json
from pathlib import Path

from src.data_splits import create_split_manifest


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    with (project_root / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    model_config = config.get("models", {})
    manifest_path = project_root / "data" / "splits.csv"
    manifest = create_split_manifest(
        processed_dir=project_root / "data" / "processed",
        output_path=manifest_path,
        test_size=model_config.get("test_size", 0.2),
        validation_size=model_config.get("validation_size", 0.1),
        random_state=model_config.get("seed", 42),
    )

    print(f"Split manifest saved to {manifest_path}")
    print(manifest.groupby(["split", "class_name"]).size().to_string())


if __name__ == "__main__":
    main()