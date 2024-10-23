import re
import matplotlib.pyplot as plt

# ログファイルを読み込む
with open('../logs/GT-D2G-ja/23/cout.txt', 'r') as file:
    log_content = file.read()

# 正規表現パターン
pattern = r'epoch#(\d+).*?train Done, avg loss=[\d.]+, train_acc=([\d.]+).*?validation Done, avg loss=[\d.]+, acc=([\d.]+)'

# データ抽出
matches = re.findall(pattern, log_content, re.DOTALL)

# データを分離
epochs = [int(match[0]) for match in matches]
train_accs = [float(match[1]) for match in matches]
accs = [float(match[2]) for match in matches]

# グラフ描画
plt.figure(figsize=(12, 6))
plt.plot(epochs, accs, label='Validation Accuracy')
plt.plot(epochs, train_accs, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

# グラフを保存
plt.savefig('accuracy_plot.png')
plt.show()

print(f"Total epochs: {len(epochs)}")
print(f"Final validation accuracy: {accs[-1]:.2f}")
print(f"Final training accuracy: {train_accs[-1]:.2f}")