import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

def get_mnist_dataloader(batch_size=64, train=True, download=True):
    """
    Returns a DataLoader for the MNIST dataset.

    Args:
        batch_size (int): Number of samples per batch.
        train (bool): If True, returns the training dataset; otherwise, returns the test dataset.
        download (bool): If True, downloads the dataset if it is not already present.

    Returns:
        DataLoader: A DataLoader for the MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_dataset = datasets.MNIST(root='./data', train=train, download=download, transform=transform)
    
    return torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

def train_mnist(epochs=1, batch_size=64, lr=0.01, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = get_mnist_dataloader(batch_size=batch_size, train=True)
    test_loader = get_mnist_dataloader(batch_size=batch_size, train=False)

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)
        def forward(self, x):
            x = x.view(-1, 28*28)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

    # テスト精度の計算
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    # Example usage
    train_loader = get_mnist_dataloader(batch_size=64, train=True)
    test_loader = get_mnist_dataloader(batch_size=64, train=False)

    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break  # Just to show the first batch
    for images, labels in test_loader:
        print(images.shape, labels.shape)
        break
    # Just to show the first batch
    print("MNIST DataLoader is ready.")
    print("\n--- Training MNIST ---")
    train_mnist(epochs=1, batch_size=64)