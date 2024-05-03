import gzip
import pickle
import itertools

import more_itertools
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import random
import japanize_matplotlib

import make_corpus_from_dataset
import phrase_extraction_by_spacy

def create_initial_graph(index, sentences, window_size: int = 4) -> nx.Graph:
    G = nx.Graph()


    G.graph['docid'] = index
    G.graph['class'] = random.randint(0, 4)

    print(len(sentences))
    nodes = []
    for sentence in sentences:
        noun_phrases = phrase_extraction_by_spacy.noun_phrases_extraction_by_spacy(sentence)
        verb_phrases = phrase_extraction_by_spacy.verb_phrases_extract_by_spacy(sentence)
        adj_phrases = phrase_extraction_by_spacy.adj_phrases_extraction_by_spacy(sentence)

        # ノード追加
        nodes += noun_phrases + verb_phrases + adj_phrases
    # すべての文のnode追加
    for i in range(len(nodes)):
        G.add_node(nodes[i])

        if i >= 2:
            # sliding-windowの設定
            sliding_window_list = more_itertools.windowed(nodes,window_size,step = 3)
            # すべての組み合わせでエッジ接続
            for window in sliding_window_list:
                for comb_nodes in itertools.combinations(window, 2):
                    # Noneの場合はエッジ接続しない
                    if comb_nodes[0] is not None and comb_nodes[1] is not None:
                        G.add_edge(comb_nodes[0],comb_nodes[1])
                    
    return G

def show_NetWorkX_graph(graph):
    G = graph
    print(nx.number_of_nodes(G))
    print(nx.number_of_edges(G))
    # グラフの描画
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, alpha=0.5, font_family='IPAexGothic')

    # 表示
    plt.axis("off")
    plt.show()

def write_pickle(graphs):
    # 名前は.gzだが、実際は圧縮されていないので注意
    with open('../data/livedoor.win5.pickle.gz', 'wb') as f:
      pickle.dump(graphs, f)


def main():
    nlp = spacy.load('ja_ginza')
    # 100万文字以上は以下の設定をする必要あり
    nlp.max_length = 1500000
    text = make_corpus_from_dataset.return_livedoor_text('https://www.rondhuit.com/download/ldcc-20140209.tar.gz',
                                                        'ldcc-20140209.tar.gz')
    graph_li = []
    for category in text.values():
        for index, content in enumerate(category):
            doc = nlp(content)
            # doc.sentsは1文ずつのオブジェクト
            file_init_graph = create_initial_graph(index, list(doc.sents))
            graph_li.append(file_init_graph)
    
    write_pickle(graph_li)

if __name__ == "__main__":
    main()
