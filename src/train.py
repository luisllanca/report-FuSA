import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SimpleMNISTModel
from pathlib import Path

def train(data_dir, model_dir, epochs=5, batch_size=64, lr=0.001):
    
    train_dataset = torch.load(f"{data_dir}/mnist_train.pt")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleMNISTModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{model_dir}/mnist_model.pth")
    print(f"Modelo guardado en {model_dir}/mnist_model.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entrenar el modelo MNIST.")
    parser.add_argument("--data_dir", type=str, required=True, help="Ruta del dataset procesado.")
    parser.add_argument("--model_dir", type=str, required=True, help="Ruta para guardar el modelo entrenado.")
    args = parser.parse_args()

    train(data_dir=args.data_dir, model_dir=args.model_dir, epochs=5)
