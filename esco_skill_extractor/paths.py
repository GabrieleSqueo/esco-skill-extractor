from pathlib import Path


def package_root() -> Path:
    """Directory containing esco_skill_extractor package resources (templates, data, …)."""
    return Path(__file__).resolve().parent


def package_data_dir() -> Path:
    return package_root() / "data"
