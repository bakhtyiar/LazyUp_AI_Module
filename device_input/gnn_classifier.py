import time
import tracemalloc

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from device_input.device_log_loader import load_device_logs


class EventGNN(nn.Module):
    def __init__(self, hidden_channels=64):
        super(EventGNN, self).__init__()
        # Учитываем 3 признака на узел: buttonKey, временные дельты, нормализованный timestamp
        self.node_encoder = nn.Linear(3, hidden_channels)

        # Два слоя GCN с активацией ReLU
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, batch):
        # Проход через GNN
        x = self.node_encoder(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Глобальный пулинг для получения graph-level представления
        x = global_mean_pool(x, batch)

        # Классификация
        return torch.sigmoid(self.classifier(x)).squeeze(1)


def prepare_graph_data(sequences, labels):
    data_list = []

    for seq, label in zip(sequences, labels):
        # Извлекаем features
        button_keys = torch.tensor([e['buttonKey'] for e in seq], dtype=torch.float)
        timestamps = [e['dateTime'] for e in seq]
        deltas = torch.tensor([(timestamps[i] - timestamps[i - 1])
                               if i > 0 else 0 for i in range(len(seq))], dtype=torch.float)
        normalized_ts = torch.tensor([(ts - timestamps[0]) for ts in timestamps], dtype=torch.float)

        # Собираем node features
        x = torch.stack([button_keys, deltas, normalized_ts], dim=1)

        # Строим полносвязный граф (можно изменить на другую топологию)
        num_nodes = len(seq)
        edge_index = torch.tensor([[i, j] for i in range(num_nodes)
                                   for j in range(num_nodes) if i != j], dtype=torch.long).t()

        # Создаем объект Data
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))
        data_list.append(data)

    return data_list


# Пример использования
if __name__ == "__main__":
    sample_data = load_device_logs(1000)

    # Подготовка данных
    X = [item['list'] for item in sample_data]
    y = [item['mode'] for item in sample_data]
    graph_data = prepare_graph_data(X, y)

    # Разделение на train/test
    train_data, test_data = train_test_split(graph_data, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=2)

    # Инициализация модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EventGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    # Измерение использования памяти до обучения
    tracemalloc.start()
    start_train = time.time()
    # Обучение
    for epoch in range(100):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Валидация
        model.eval()
        correct = 0
        for data in test_loader:
            data = data.to(device)
            pred = (model(data.x, data.edge_index, data.batch) > 0.5).float()
            correct += (pred == data.y).sum().item()
        acc = correct / len(test_data)

        print(f'Epoch {epoch}, Loss: {total_loss:.4f}, Test Acc: {acc:.4f}')

    end_train = time.time()
    training_time = end_train - start_train
    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()
    print(classification_report(y_test, y_pred))
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Inference time: {inference_time:.4f} s")
