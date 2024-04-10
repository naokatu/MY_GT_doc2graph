import pickle
import networkx as nx
import matplotlib.pyplot as plt

# G = nx.Graph()

with open('../data/dblp.win5.pickle.gz', 'rb')as f:
  all_graphs = pickle.load(f)
  # グラフ数の取得
  lens = len(all_graphs)



G = all_graphs[0]  # 最初のグラフを取得
print(nx.number_of_nodes(G))
print(nx.number_of_edges(G))
# グラフの描画
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos, with_labels=True, alpha=0.5)

# 表示
plt.axis("off")
plt.show()
