from pathlib import Path

import pandas as pd


def load_skills_table(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "skills.csv")


def load_occupations_table(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "occupations.csv")
