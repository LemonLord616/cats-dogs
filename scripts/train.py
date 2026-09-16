import torch
import torch.nn as nn

from cats_dogs.model import CNNClassifier, ImageDataset

from config import RunParameters, settings


def train(params: RunParameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    dataset = ImageDataset(
        settings.train_file,
        data_dir=settings.out_dir,
        image_size=params.image_size,
        augment=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        pin_memory=True,
    )

    losses = []
    for epoch in range(params.epochs):
        model.train()
        total_loss = 0.0
        for images, y in dataloader:
            images, y = images.to(device), y.to(device)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), settings.model_dir / params.model_name)

    return losses


if __name__ == "__main__":
    train(RunParameters())
