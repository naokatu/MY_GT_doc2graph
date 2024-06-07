import pickle
import networkx as nx
import matplotlib.pyplot as plt
import pprint
import numpy as np
import gzip
import japanize_matplotlib

# G = nx.Graph()

with open('../data/nyt.win5.pickle.gz', 'rb')as f:
  all_graphs = pickle.load(f)
  print(type(all_graphs))
  # グラフ数の取得
  lens = len(all_graphs)

G = all_graphs[23]  # 最初のグラフを取得

# print(nx.number_of_nodes(G))
# print(nx.number_of_edges(G))
node_li = list(G.nodes)
print(G.graph)
print(*node_li)
# グラフの描画
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos, with_labels=True, alpha=0.5, font_family='IPAexGothic')

# 表示
plt.axis("off")
plt.show()
