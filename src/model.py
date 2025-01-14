import torch.nn as nn
import torch.nn.functional as F

class SimpleMNISTModel(nn.Module):
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Primera capa completamente conectada
        self.fc2 = nn.Linear(128, 64)       # Segunda capa completamente conectada
        self.fc3 = nn.Linear(64, 10)        # Capa de salida con 10 clases (dígitos 0-9)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Aplanar la imagen 28x28 a un vector
        x = F.relu(self.fc1(x))  # Activación ReLU después de la primera capa
        x = F.relu(self.fc2(x))  # Activación ReLU después de la segunda capa
        x = self.fc3(x)          # Salida sin activación, para usar en CrossEntropyLoss
        return x
