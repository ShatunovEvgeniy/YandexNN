from clearml import Dataset
import torch
import pandas as pd


dataset = Dataset.create(dataset_project="SimpleNN Project", dataset_name="synthetic_data")

# Генерируем случайные данные
torch.manual_seed(0)
X = torch.randn(3, 5)   # форма [3, 5]
y = torch.randn(3, 1)   # форма [3, 1]

df = pd.DataFrame(
    X.numpy(),
    columns=[f"f{i+1}" for i in range(5)]   # f1, f2, f3, f4, f5
)
df["label"] = y.numpy()  # добавляем колонку "label"

df.to_csv("data.csv", index=False)
dataset.add_files("./data.csv")
dataset.upload()
dataset.finalize()