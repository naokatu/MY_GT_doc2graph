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

def create_initial_graph(num_category, index, sentences, window_size: int = 2) -> nx.Graph:
    G = nx.Graph()

    G.graph['docid'] = index
    G.graph['class'] = num_category

    nodes = []
    noun_offsets_li = []
    verb_offsets_li = []
    adj_offsets_li = []
    for sentence in sentences:
        noun_phrases, noun_offsets = phrase_extraction_by_spacy.noun_phrases_extraction_by_spacy(sentence)
        verb_phrases, verb_offsets = phrase_extraction_by_spacy.verb_phrases_extract_by_spacy(sentence)
        adj_phrases, adj_offsets = phrase_extraction_by_spacy.adj_phrases_extraction_by_spacy(sentence)

        # ノード追加
        nodes += noun_phrases + verb_phrases + adj_phrases
        noun_offsets_li += noun_offsets
        verb_offsets_li += verb_offsets
        adj_offsets_li += adj_offsets
    # すべての文のnode追加
    for i in range(len(nodes)):
        G.add_node(nodes[i])
        # 単語の出現頻度を計算
        if 'freq' not in G.nodes[nodes[i]]:
            G.nodes[nodes[i]]['freq'] = 1
        else:
            G.nodes[nodes[i]]['freq'] += 1

        # 単語の出現位置を追加
        # TODO:本当は各offsetsが空かどうか判定する必要あり
        if len(noun_offsets_li) - 1 >= i:
            if 'offsets' not in G.nodes[nodes[i]]:
                G.nodes[nodes[i]]['offsets'] = [noun_offsets_li[i]]
            else:
                G.nodes[nodes[i]]['offsets'].append(noun_offsets_li[i])
        elif len(noun_offsets_li) + len(verb_offsets_li) - 1 >= i:
            if 'offsets' not in G.nodes[nodes[i]]:
                G.nodes[nodes[i]]['offsets'] = [verb_offsets_li[i-len(noun_offsets_li)]]
            else:
                G.nodes[nodes[i]]['offsets'].append(verb_offsets_li[i-len(noun_offsets_li)])
        else:
            if 'offsets' not in G.nodes[nodes[i]]:
                G.nodes[nodes[i]]['offsets'] = [adj_offsets_li[i-len(noun_offsets_li)-len(verb_offsets_li)]]
            else:
                G.nodes[nodes[i]]['offsets'].append(adj_offsets_li[i-len(noun_offsets_li)-len(verb_offsets_li)])

        if i >= 2:
            # sliding-windowの設定
            sliding_window_list = more_itertools.windowed(nodes,window_size,step = 1)
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
    print(text.keys()) # dict_keys(['text/it-life-hack', 'text/movie-enter', 'text/livedoor-homme', 'text/smax', 'text/topic-news', 'text/kaden-channel', 'text/sports-watch', 'text/dokujo-tsushin', 'text/peachy'])
    for num_category, category in enumerate(text.keys()):
        print(category)
        for index, content in enumerate(text[category]):
            doc = nlp(content)
            # doc.sentsは1文ずつのオブジェクト
            file_init_graph = create_initial_graph(num_category, index, list(doc.sents))
            graph_li.append(file_init_graph)
    
    write_pickle(graph_li)

if __name__ == "__main__":
    main()
