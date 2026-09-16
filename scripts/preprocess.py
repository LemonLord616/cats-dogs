import csv

from PIL import Image
from sklearn.model_selection import train_test_split

from config import RunParameters, settings

LABELS = {"Cat": 0, "Dog": 1}


def is_valid_image(path):
    try:
        Image.open(path).convert("RGB")
        return True
    except Exception:
        return False


def collect_samples():
    paths, labels = [], []
    for label_name, label_id in LABELS.items():
        for img_path in (settings.raw_dir / label_name).iterdir():
            if is_valid_image(img_path):
                paths.append(str(img_path))
                labels.append(label_id)
    return paths, labels


def preprocess(params: RunParameters):
    paths, labels = collect_samples()
    X_train, X_test, y_train, y_test = train_test_split(
        paths,
        labels,
        test_size=params.test_size,
        random_state=params.random_seed,
        stratify=labels,
    )

    settings.out_dir.mkdir(parents=True, exist_ok=True)
    for name, X, y in (
        (settings.train_file, X_train, y_train),
        (settings.test_file, X_test, y_test),
    ):
        with open(settings.out_dir / name, "w") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(zip(X, y))


if __name__ == "__main__":
    preprocess(RunParameters())
