from datetime import datetime

import mlflow
import torch

from config import RunParameters, settings
from train import train
from evaluate import evaluate
from cats_dogs.model import CNNClassifier


def _unique_model_name() -> str:
    return f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"


def _log_params(params: RunParameters) -> None:
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    mlflow.log_params(
        {
            "model_name": params.model_name,
            "epochs": params.epochs,
            "learning_rate": params.learning_rate,
            "batch_size": settings.batch_size,
            "image_size": params.image_size,
            "num_workers": settings.num_workers,
            "test_size": params.test_size,
            "random_seed": params.random_seed,
            "device": device,
        }
    )


def _log_model(params: RunParameters) -> None:
    model_path = settings.model_dir / params.model_name
    mlflow.log_artifact(model_path)

    model = CNNClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    example = torch.randn(1, 3, params.image_size, params.image_size)
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        input_example=example,
        serialization_format="pickle",
    )


def run(params: RunParameters) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run():
        _log_params(params)
        for step, loss in enumerate(train(params)):
            mlflow.log_metric("train_loss", loss, step=step)
        _log_model(params)
        mlflow.log_metrics(evaluate(params))


if __name__ == "__main__":

    random_seed = 42

    epochs = [4, 6]
    learning_rate = [1e-3, 1e-2]
    image_size = [64, 128]
    test_size = [0.2, 0.3]
    for e in epochs:
        for lr in learning_rate:
            for i_s in image_size:
                for t_s in test_size:
                    run(
                        RunParameters(
                            e, lr, random_seed, i_s, t_s, _unique_model_name()
                        )
                    )
