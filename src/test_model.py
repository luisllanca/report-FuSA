import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SimpleMNISTModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def test(data_dir, model_path, metrics_path, plot_path):
    test_dataset = torch.load(f"{data_dir}/mnist_test.pt")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleMNISTModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_labels = []
    all_predictions = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    with open(metrics_path, "w") as f:
        f.write(f"accuracy: {accuracy:.2f}\n")

    cm = confusion_matrix(all_labels, all_predictions, labels=np.arange(10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(plot_path)
    print(f"Gráfico guardado en: {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prueba el modelo MNIST y guarda métricas y gráficos.")
    parser.add_argument("--data_dir", type=str, required=True, help="Ruta del dataset procesado.")
    parser.add_argument("--model_path", type=str, required=True, help="Ruta del modelo entrenado.")
    parser.add_argument("--metrics_path", type=str, required=True, help="Ruta para guardar las métricas.")
    parser.add_argument("--plot_path", type=str, required=True, help="Ruta para guardar el gráfico.")
    args = parser.parse_args()

    test(data_dir=args.data_dir, model_path=args.model_path, metrics_path=args.metrics_path, plot_path=args.plot_path)
