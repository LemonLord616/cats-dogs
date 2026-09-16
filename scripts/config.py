from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )

    raw_dir: Path = Path("data/raw/PetImages")
    out_dir: Path = Path("data/processed")
    model_dir: Path = Path("data/models")

    train_file: str = "train.csv"
    test_file: str = "test.csv"

    batch_size: int = 128
    num_workers: int = 4

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "cats_dogs"


settings = Settings()


@dataclass
class RunParameters:
    epochs: int = 10
    learning_rate: float = 1e-3
    random_seed: int = 42
    image_size: int = 128
    test_size: float = 0.2
    model_name: str = "model.pt"
