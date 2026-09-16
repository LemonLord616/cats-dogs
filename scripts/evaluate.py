import torch
import torch.nn as nn

from cats_dogs.model import CNNClassifier, ImageDataset

from config import settings, RunParameters


def evaluate(params: RunParameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNClassifier().to(device)
    model.load_state_dict(
        torch.load(
            settings.model_dir / params.model_name,
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    dataset = ImageDataset(
        settings.test_file,
        data_dir=settings.out_dir,
        image_size=params.image_size,
        augment=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
    )

    criterion = nn.BCEWithLogitsLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, y in dataloader:
            images, y = images.to(device), y.to(device)
            outputs = model(images).squeeze(1)
            total_loss += criterion(outputs, y).item() * y.size(0)

            preds = (torch.sigmoid(outputs) >= 0.5).int()
            correct += (preds == y.int()).sum().item()
            total += y.size(0)

    print(f"Test Loss: {total_loss / total:.4f}")
    print(f"Test Accuracy: {correct / total * 100:.2f}%")
    return {
        "test_loss": total_loss / total,
        "test_accuracy": correct / total,
    }


if __name__ == "__main__":
    evaluate(RunParameters())
