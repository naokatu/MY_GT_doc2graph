import torch
import torch.nn as nn
import torch.optim as optim

# GPUが使用可能かを確認
print("GPU available:", torch.cuda.is_available())

# サンプルデータの準備
x_train = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
y_train = [[0.4], [0.5], [0.6], [0.7]]

# サンプルデータをPyTorchのテンソルに変換
x_train = torch.tensor(x_train, dtype=torch.float32).view(-1, 1, 3)
y_train = torch.tensor(y_train, dtype=torch.float32)

# デバイスの設定（GPUを使用する場合）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = x_train.to(device)
y_train = y_train.to(device)

# RNNモデルの定義
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1, :])
        return y_pred

# モデルのインスタンス化
model = RNNModel(input_size=3, hidden_size=10, output_size=1).to(device)

# 損失関数とオプティマイザの定義
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# モデルの訓練
num_epochs = 100
for epoch in range(num_epochs):
    # 順伝播
    y_pred = model(x_train)

    # 損失の計算
    loss = criterion(y_pred, y_train)

    # 勾配のゼロ化、逆伝播、パラメータの更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 10エポックごとに損失を表示
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
