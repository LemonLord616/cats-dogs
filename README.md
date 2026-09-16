# Cats vs Dogs Classification App



## Install all dependencies (not required).
```
uv sync --all-extras
source .venv/bin/activate
```

## Run app
All essential parameters are stored in .env file. You can run application using these commands:
```
cp .env.example .env
docker compose --env-file .env up --build -d
```
You can test this classification app without training the model. This repository already includes `model.pt` at `data/models/model.pt`, which is already configured to be used in `.env.example`


## Train dependencies, Kaggle credentials, DVC
Optionally add train optional dependencies
```
uv sync --extra train
source .venv/bin/activate
```

Make sure you have `~/.kaggle/kaggle.json`, because dataset is located at Kaggle, and Kaggle requires auth. Then you can execute this to download and prepare your data:
```
dvc repro
```
Datasets will be downloaded ad data/raw/ directory and processed data will be saved at data/processed/ directory.
Then you can try:
```
uv run scripts/flow.py
```
This will run simple hyperparameter sweep using MLFlow. To run MLFlow UI use:
```
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
MLFlow UI is accesible at `http://localhost:5000`.

## Structure

| Path | Description |
| -------- | -------- |
| `scripts/{preprocess.py, train.py, evaluate.py}` | Core parts of training pipeline, you can reuse according methods to make up your own training pipeline |
| `scripts/flow.py` | Example of MLFlow usage, runs simple hyperparameter sweep. MLFlow UI is accesible at `http://localhost:5000` |
| `.env.example` | Example configuration |
| `services/{api, app}` | Model API and app respectively. Model API is configured to use `model.pt` at `data/models` by default, this repository provides this model, so you can easily try it out. |
| `dvc.yaml` | Basic dataset pipeline is configured here |

All scripts are ran using `uv`:
```
uv run scripts/script.py # e.g. uv run scripts/train.py
```
