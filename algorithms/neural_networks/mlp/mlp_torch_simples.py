import torch
import torch.nn as nn
import torch.optim as optim

# Verifica se o dispositivo MPS está disponível
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Dados de entrada (função XOR)
X = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=torch.float32).to(device)

# Saídas desejadas
y = torch.tensor([
    [0],
    [1],
    [1],
    [0]
], dtype=torch.float32).to(device)

# Definição do modelo
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        # Definindo as camadas
        self.layer1 = nn.Linear(2, 2)  # Camada oculta com 2 neurônios
        self.layer2 = nn.Linear(2, 1)  # Camada de saída

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # Ativação ReLU na camada oculta
        x = torch.sigmoid(self.layer2(x))  # Ativação sigmoid na camada de saída
        return x

# Instancia o modelo e move para o dispositivo
model = XORModel().to(device)

# Define a função de perda e o otimizador
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Treinamento do modelo
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass e otimização
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Impressão do erro a cada 1000 épocas
    if (epoch+1) % 1000 == 0:
        print(f'Época [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Teste do modelo
with torch.no_grad():
    outputs = model(X)
    predicted = (outputs > 0.5).float()
    print('\nResultados:')
    for i in range(len(X)):
        print(f'Entrada: {X[i].cpu().numpy()}, Saída Prevista: {predicted[i].item()}')