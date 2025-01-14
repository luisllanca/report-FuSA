import torch
from torchvision import datasets, transforms
from pathlib import Path
import argparse

def process_mnist(input_dir, output_dir):
    """
    Procesa el dataset MNIST ya descargado y aplica transformaciones.
    """
    # Transformaciones
    transform = transforms.Compose([
        transforms.RandomRotation(30),  # Rotar imágenes aleatoriamente
        transforms.RandomHorizontalFlip(),  # Flip horizontal aleatorio
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalización
    ])

    # Cargar el dataset desde los archivos existentes
    train_dataset = datasets.MNIST(root=input_dir, train=True, transform=transform, download=False)
    test_dataset = datasets.MNIST(root=input_dir, train=False, transform=transform, download=False)

    # Crear directorio de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Guardar los datos procesados
    torch.save(train_dataset, f"{output_dir}/mnist_train.pt")
    torch.save(test_dataset, f"{output_dir}/mnist_test.pt")
    print(f"Dataset procesado guardado en: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar el dataset MNIST.")
    parser.add_argument("--input_dir", type=str, required=True, help="Ruta de entrada del dataset MNIST.")
    parser.add_argument("--output_dir", type=str, required=True, help="Ruta donde se guardará el dataset procesado.")
    args = parser.parse_args()
    process_mnist(args.input_dir, args.output_dir)
