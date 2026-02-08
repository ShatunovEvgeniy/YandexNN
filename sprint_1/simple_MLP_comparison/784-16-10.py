
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from clearml import Task, Logger
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Название модели для отображения графиков и логирования
    name = 'MLP_1layer'

    # Новый эксперимент в ClearML
    task = Task.init(project_name='NeuralNetworkCourse',
                    task_name=name,
                    reuse_last_task_id=False)
    logger = Logger.current_logger()


    # Подготовка данных MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset,   batch_size=1000, shuffle=False)

    device = torch.device('cuda')

    # Модель 784→16→10
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 16),
        nn.ReLU(),
        nn.Linear(16, 10)
    ).to(device)

    num_epochs = 5
    hist = {name: {'train_loss': [], 'val_loss': [], 'val_acc': []} }

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(1, num_epochs + 1):
        # Обучение
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        # Добавляем потерю для графика
        hist[name]['train_loss'].append(epoch_train_loss)


        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                val_loss += loss.item()
                preds = output.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        # Валидация на каждой эпохе обучения
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc  = 100.0 * correct / total
        # Добавляем потерю для графика
        hist[name]['val_loss'].append(epoch_val_loss)
        # Добавляем точность для графика
        hist[name]['val_acc'].append(epoch_val_acc)


        # Логирование в ClearML 
        logger.report_scalar("Loss", "Train", epoch_train_loss, epoch)
        logger.report_scalar("Loss", "Validation", epoch_val_loss, epoch)
        logger.report_scalar("Accuracy", "Validation", epoch_val_acc, epoch)


        print(f"{name} Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {epoch_train_loss:.4f}, "
                f"Val Loss: {epoch_val_loss:.4f}, "
                f"Val Acc: {epoch_val_acc:.2f}%")


    task.close()


    # Визуализация истории обучения (пример для loss)
    plt.figure(figsize=(12, 6))

    # Потеря
    plt.subplot(1, 2, 1)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, hist[name]['val_loss'],   marker='x', label=f"{name} Val Loss")
    plt.plot(epochs, hist[name]['train_loss'], marker='o', label=f"{name} Train Loss",)
    plt.title("Сравнение кривых потерь")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Точность
    plt.subplot(1, 2, 2)

    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, hist[name]['val_acc'], marker='o', label=f"{name} Val Acc")
    plt.title("Сравнение точности на валидации")
    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()