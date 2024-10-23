import pickle
import networkx as nx
import matplotlib.pyplot as plt
import pprint
import numpy as np
import gzip
import japanize_matplotlib

# G = nx.Graph()

with open('../data/livedoor.win5.pickle.gz', 'rb')as f:
  all_graphs = pickle.load(f)
  print(type(all_graphs))
  # グラフ数の取得
  lens = len(all_graphs)


G = all_graphs[5920]  # 最初のグラフを取得
print(lens)
# print(nx.number_of_nodes(G))
# print(nx.number_of_edges(G))
node_li = list(G.nodes)
print(G.graph)
print(*node_li)
for i in node_li:
  print(G.nodes[i])
  print(i)

# グラフの描画
pos = nx.spring_layout(G)
plt.figure(figsize=(15,15))
nx.draw_networkx(G, pos, with_labels=True, alpha=0.5, font_family='IPAexGothic')

# 表示
plt.axis("off")
plt.show()
