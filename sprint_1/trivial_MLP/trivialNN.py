from clearml import Task, Dataset
import pandas as pd
import torch
from torch import nn
import os


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=5, out_features=5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Инициализация ClearML-задачи
    task = Task.init(project_name="SimpleNN Project", task_name="Experiment #1")
    config = {"learning_rate": 0.01, "batch_size": 4, "epochs": 100}
    task.connect(config)

    # Создание модели, оптимизатора и функции потерь
    device = torch.device('cuda')
    model = SimpleNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    # Загрузка данных
    dataset = Dataset.get(dataset_project="SimpleNN Project", dataset_name="synthetic_data")
    data_path = dataset.get_local_copy()  # ← здесь получится путь к папке с CSV

    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    csv_path = os.path.join(data_path, csv_files[0])

    df = pd.read_csv(csv_path)
    X_np = df.iloc[:, :-1].values  # все столбцы, кроме последнего → (N, 5)
    y_np = df.iloc[:, -1:].values  # последний столбец как (N, 1)
    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    y = torch.tensor(y_np, dtype=torch.float32).to(device)

    # Обучение модели
    for epoch in range(config["epochs"]):
        outputs = model(X)
        loss = criterion(outputs, y)

        # Шаг обучения
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Логируем loss через task с указанием итерации
        task.get_logger().report_scalar("Loss", "train", loss.item(), epoch)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Epoch {epoch}, Loss: {loss.item():.4f}")