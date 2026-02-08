import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from clearml import Task, Logger


class AdvancedMNISTMLPList(nn.Module):
    def __init__(self):
        super(AdvancedMNISTMLPList, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ])

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # картинка в вектор
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda")
    model = AdvancedMNISTMLPList().to(device)

    # Загрузка данных MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # Инициализация функции потерь и оптимизатора
    criterion =  nn.CrossEntropyLoss()
    optimizer =  optim.Adam(model.parameters(), lr=0.001)

    # Подключение к ClearML
    task = Task.init(project_name='MNIST_Project', task_name='MLP_Train')

    # Цикл обучения
    num_epochs =  10
    train_losses = []
    train_accuracies = []

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Проход по всем батчам трен.набора
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Прямой проход
            preds = model(images)
            loss = criterion(preds, labels)

            # Обратный проход и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Накопление статистики для метрик
            running_loss += loss.item()
            total += labels.size(0)
            _, predicted = preds.max(1)
            correct += predicted.eq(labels).sum().item()

        # Средние значения по эпохе
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f'Epoch {epoch + 1}/{num_epochs} — Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        # Отправка метрик в ClearML
        Logger.current_logger().report_scalar("train", "loss", iteration=epoch, value=epoch_loss)
        Logger.current_logger().report_scalar("train", "accuracy", iteration=epoch, value = epoch_acc)

    # Оценка на тестовом наборе
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)

            test_loss += loss.item()
            total += labels.size(0)
            _, predicted = preds.max(1)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_accuracy = 100.0 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # Отправка тестовых метрик в ClearML
    Logger.current_logger().report_scalar("test", "loss", iteration=num_epochs, value=test_loss)
    Logger.current_logger().report_scalar("test", "accuracy", iteration=num_epochs, value=test_accuracy)

    # Построение графиков потерь и точности
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', color='blue', label='Train Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Эпоха')
    plt.ylabel('Потеря')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, marker='o', color='green', label='Train Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()