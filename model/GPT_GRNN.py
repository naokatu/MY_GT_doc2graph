"""
Graph Pointer Network for node proposal, Graph RNN for edge proposal
Author:
Create Date: Nov 27, 2020
"""

from pprint import pprint
import math
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import AvgPooling, GlobalAttentionPooling
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from allennlp.nn.util import masked_softmax, get_mask_from_sequence_lengths


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    import torch.nn.init as init
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class GCNEncoder(nn.Module):
    def __init__(self, pretrained_emb: th.Tensor, GCN_in_size: int, GCN_hidden_size: int,
                 pooling: str = 'mean', yelp_senti_feat: bool = False,
                 pretrained_emb_dropout: float = 0.0):
        super(GCNEncoder, self).__init__()
        self.yelp_senti_feat = yelp_senti_feat
        self.pretrained_emb_dropout = pretrained_emb_dropout
        # 単語埋め込みを使用してnn.Embedding（各単語のインデックスに意味を持たせるベクトル）を初期化
        self.word_emb = nn.Embedding.from_pretrained(pretrained_emb, padding_idx=1)
        if self.yelp_senti_feat:
            sentiment_level = 3    # currently fix
            ee_dim = 2             # currently fix
            self.pos_ee = nn.Embedding(sentiment_level, ee_dim)
            self.neg_ee = nn.Embedding(sentiment_level, ee_dim)
            GCN_in_size += (2 * ee_dim)
        if self.pretrained_emb_dropout > 0.0:
            self.dropout = nn.Dropout(pretrained_emb_dropout)
        # グラフ畳み込み層の初期化　GCN_in_size→GCN_hidden_size GCN_hidden_size→GCN_hidden_size
        self.conv1 = GraphConv(GCN_in_size, GCN_hidden_size, allow_zero_in_degree=True)
        self.conv2 = GraphConv(GCN_hidden_size, GCN_hidden_size, allow_zero_in_degree=True)
        # 指定されたプーリング方法に基づいてプーリング層を初期化　初期値は平均プーリング
        if pooling == 'mean': # 平均プーリング
            self.pooling = AvgPooling()
        elif pooling == 'global_attention':
            pooling_gate_nn = nn.Linear(GCN_hidden_size, 1)
            self.pooling = GlobalAttentionPooling(pooling_gate_nn) # グローバル注意プーリング
        else:
            print('pooling is invalid!!')
            exit(-1)
    
    # 単語埋め込みの計算
    def forward(self, g: dgl.DGLGraph) -> tuple:
        word_embs = self.word_emb(g.ndata['p'])   # nnodes*mention_l*emb_s
        word_embs = th.div(th.sum(word_embs, dim=1), g.ndata['ml'].unsqueeze(1))   # nnodes*emb_s
        if self.pretrained_emb_dropout:
            word_embs = self.dropout(word_embs)
        # 単語埋め込みとノード特徴量の連結　ノードのテキスト情報をベクトルに組み込める
        if not self.yelp_senti_feat:
            h = th.cat((word_embs, g.ndata['f'].unsqueeze(1),
                        g.ndata['lf'].unsqueeze(1), g.ndata['ll'].unsqueeze(1)), 1)  # nnodes*(emb_s+3)
        else:
            pos_embs = self.pos_ee(g.ndata['pos'])   # nnodes*ee_dim
            neg_embs = self.neg_ee(g.ndata['neg'])   # nnodes*ee_dim
            h = th.cat((word_embs, g.ndata['f'].unsqueeze(1), g.ndata['lf'].unsqueeze(1),
                        g.ndata['ll'].unsqueeze(1), pos_embs, neg_embs), 1)  # nnodes*(emb_s+3+2*ee_dim)

        h = F.relu(self.conv1(g, h))     # nnodes*gcn_h
        h = F.relu(self.conv2(g, h))     # nnodes*gcn_h
        hg = self.pooling(g, h)    # batch*gcn_h
        return h, hg


class BaselineClassifier(nn.Module):
    """
    GCNEncoder + MLP serving as baseline model
    """
    def __init__(self, pretrained_emb: th.Tensor, GCN_in_size: int, GCN_hidden_size: int,
                 n_labels: int, pooling: str = 'mean', yelp_senti_feat: bool = False):
        super(BaselineClassifier, self).__init__()
        self.gcn = GCNEncoder(pretrained_emb, GCN_in_size, GCN_hidden_size, pooling, yelp_senti_feat)
        self.mlp = nn.Sequential(
                nn.Linear(GCN_hidden_size, GCN_hidden_size//4),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(GCN_hidden_size//4, n_labels)
                )

    def forward(self, g: dgl.DGLGraph) -> th.Tensor:
        h, hg = self.gcn(g)
        pred = self.mlp(hg)   # batch*n_labels
        return pred


class Attention(nn.Module):
    def __init__(self, hidden_size, units):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_size, units, bias=False)
        self.W2 = nn.Linear(hidden_size, units, bias=False)
        self.wc = nn.Linear(1, units, bias=False)
        self.V = nn.Linear(units, 1, bias=False)

    def forward(self,
                encoder_out: th.Tensor,
                decoder_hidden: th.Tensor,
                cov_vec: th.Tensor,
                mask: th.BoolTensor):
        # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
        # decoder_hidden: (BATCH, HIDDEN_SIZE)
        # cov_vec: (BATCH, ARRAY_LEN)

        # uj: (BATCH, ARRAY_LEN, ATTENTION_UNITS)
        # Note: we can add the both linear outputs thanks to broadcasting
        uj = self.W1(encoder_out) + self.W2(decoder_hidden.unsqueeze(1)) + self.wc(cov_vec.unsqueeze(-1))
        uj = self.V(th.tanh(uj)).squeeze(-1)
        # uj: (BATCH, ARRAY_LEN)

        # Attention mask over inputs
        # aj: (BATCH, ARRAY_LEN)
        # aj = masked_log_softmax(uj.squeeze(-1), mask, dim=-1)
        aj = masked_softmax(uj, mask, dim=-1, memory_efficient=True)
        return aj


class GPTDecoder(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 attention_units: int = 10,
                 tau: int = 3):
        super(GPTDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.tau = tau
        self.attention = Attention(hidden_size, attention_units)
        self.gru = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, g: dgl.DGLGraph, h: th.Tensor, hg: th.Tensor,
                max_out_seq_len: int) -> tuple:
        # h:  nnodes * hidden_s
        # hg: batch * hidden_s
        pointer_argmaxs = []  # store generated sequences
        pseduo_seqs = []   # regard nodes of graph as a sequence
        nidx = 0
        for nnode in g.batch_num_nodes():
            pseduo_seqs.append(h[nidx:nidx+nnode, :])
            nidx += nnode
        mask = get_mask_from_sequence_lengths(g.batch_num_nodes(), g.batch_num_nodes().max())
        encoder_out = pad_sequence(pseduo_seqs, batch_first=True)  # batch*max_nnode*hi
        batch_size = encoder_out.shape[0]

        decoder_input = encoder_out.new_zeros((batch_size, self.hidden_size))  # batch*hi
        decoder_hidden = hg
        cov_vec = encoder_out.new_zeros((batch_size, encoder_out.shape[1]))    # batch*max_nnode
        cov_loss = cov_vec.sum()
        for i in range(max_out_seq_len):
            decoder_hidden = self.gru(decoder_input, decoder_hidden)
            att_scores = self.attention(encoder_out, decoder_hidden, cov_vec, mask)
            cov_loss += th.minimum(att_scores, cov_vec).sum()
            cov_vec = cov_vec + att_scores
            # print('att_scores:', att_scores.shape, att_scores)
            att_argmax = F.gumbel_softmax((att_scores + 1e-12).log(),
                                          tau=self.tau, hard=True, dim=1)   # batch*max_nnode
            # print('att_argmax', att_argmax.shape, att_argmax)
            # att_argmax = th.argmax(att_scores, dim=1, keepdim=True)  # (batch,), not differentiable
            # index_tensor = att_argmax.unsqueeze(-1).expand(batch_size, 1, self.hidden_size)
            # decoder_input = th.gather(encoder_out, dim=1, index=index_tensor).squeeze(1)
            decoder_input = th.matmul(encoder_out.transpose(1, 2), att_argmax.unsqueeze(-1)).squeeze(-1)
            # print(decoder_input.shape)
            pointer_argmaxs.append(att_argmax)
        # ret pointer_argmaxs: batch*max_nnode*max_out_seq_len
        return th.stack(pointer_argmaxs, dim=2), cov_loss, encoder_out


class AdjMatrixGenerator(nn.Module):
    """
    The Interaction generator in doc2graph paper
    """
    def __init__(self, input_size: int, node_size: int):
        super(AdjMatrixGenerator, self).__init__()
        out_size = node_size * node_size
        hidden_size = input_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_size)

    def forward(self, hg: th.Tensor) -> tuple:
        # hg: batch * hidden
        m = F.relu(self.linear1(hg))
        m = th.sigmoid(self.linear2(m))  # node*node, full matrix
        return m


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight.data)
        # if self.bias is not None:
        #     self.bias.data.fill_(0.0)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = th.matmul(inputs, self.weight)
        output = th.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GraphClassifier(nn.Module):
    """
    Using Simple GCN layer
    """
    def __init__(self, in_features, out_features, n_labels, bias=False):
        super(GraphClassifier, self).__init__()
        self.conv1 = GraphConvolution(in_features, out_features, bias)
        self.conv2 = GraphConvolution(out_features, out_features, bias)
        self.mlp = nn.Sequential(
                nn.Linear(out_features, out_features//4),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(out_features//4, n_labels)
                )

    def forward(self, inputs, adj):
        output = F.relu(self.conv1(inputs, adj))
        output = F.relu(self.conv2(output, adj))  # batch*seq_len*hid
        # print(f'output:{output.shape}')
        output = output.mean(dim=1)  # batch*hid
        # print(f'output_mean:{output.shape}')
        output = self.mlp(output)
        # print(f'output_mlp:{output.shape}')
        return output


class GPTGRNNDecoderS(nn.Module):
    """
    `GT-D2G-neigh` in paper

    input_size eqauls to GCN emb_size +  adj_vec size
    hidden_size equals to GCN emb_size
    """
    def __init__(self,
                 encoder_hidden_size: int,
                 attention_units: int,
                 max_out_node_size: int,
                 tau: int,
                 p_dropout: int = 0.0):
        super(GPTGRNNDecoderS, self).__init__()
        self.hidden_size = encoder_hidden_size
        self.max_out_node_size = max_out_node_size
        self.input_size = encoder_hidden_size + max_out_node_size - 1
        self.tau = tau
        self.p_dropout = p_dropout
        self.gru = nn.GRUCell(self.input_size, self.hidden_size)
        if p_dropout > 0.0:
            self.dropout = nn.Dropout(p_dropout)
        self.attention = Attention(self.hidden_size, attention_units)
        self.fout = nn.Sequential(
                nn.Linear(self.hidden_size, self.max_out_node_size - 1),
                nn.Sigmoid()
                )

    def forward(self, g: dgl.DGLGraph, h: th.Tensor, hg: th.Tensor) -> tuple:
        # h:  nnodes * hidden_s
        # hg: batch * hidden_s
        pointer_argmaxs = []  # store generated sequences
        adj_vecs = []         # store generated adj_vecs
        pseduo_seqs = []   # regard nodes of graph as a sequence
        nidx = 0
        for nnode in g.batch_num_nodes():
            pseduo_seqs.append(h[nidx:nidx+nnode, :])
            nidx += nnode
        mask = get_mask_from_sequence_lengths(g.batch_num_nodes(), g.batch_num_nodes().max())
        encoder_out = pad_sequence(pseduo_seqs, batch_first=True)  # batch*max_nnode*hi
        batch_size = encoder_out.shape[0]

        decoder_input = encoder_out.new_zeros((batch_size, self.input_size))   # batch*input
        decoder_hidden = hg
        cov_vec = encoder_out.new_zeros((batch_size, encoder_out.shape[1]))    # batch*max_nnode
        cov_loss = cov_vec.sum()
        for i in range(self.max_out_node_size):
            decoder_hidden = self.gru(decoder_input, decoder_hidden)     # batch*hidden
            if self.p_dropout > 0.0:       # add dropout to avoid overfitting
                decoder_hidden = self.dropout(decoder_hidden)
            att_scores = self.attention(encoder_out, decoder_hidden, cov_vec, mask)
            # decode node
            cov_loss += th.minimum(att_scores, cov_vec).sum()
            cov_vec = cov_vec + att_scores
            att_argmax = F.gumbel_softmax((att_scores + 1e-12).log(),
                                          tau=self.tau, hard=True, dim=1)   # batch*max_nnode
            chosen_node_emb = th.matmul(encoder_out.transpose(1, 2),
                                        att_argmax.unsqueeze(-1)).squeeze(-1)  # batch*hidden
            pointer_argmaxs.append(att_argmax)
            # decode adj vector
            if i < 1:   # i==0, not generate
                theta = encoder_out.new_zeros((batch_size, self.max_out_node_size-1))  # batch*(max_out_seq_len-1)
            else:
                theta = self.fout(decoder_hidden)       # batch*max_out, using soft weight for now
                # vec_lens = th.tensor([i for _ in range(batch_size)])
                vec_lens = g.batch_num_nodes().new_tensor([i for _ in range(batch_size)])
                theta_mask = get_mask_from_sequence_lengths(vec_lens, self.max_out_node_size-1)  # batch*max_out
                theta = theta * theta_mask
            adj_vecs.append(theta)
            decoder_input = th.cat((chosen_node_emb, theta), dim=1)    # batch*input
        # pointer_argmaxs: batch*max_nnode*max_seq_len, cov_loss: scalar
        # encoder_out: batch*max_nnode*hidden, adj_vecs:batch*max_seq-1*max_seq_len
        cov_loss = cov_loss / batch_size
        return th.stack(pointer_argmaxs, dim=2), cov_loss, encoder_out, th.stack(adj_vecs, dim=2)


class GPTGRNNDecoder(nn.Module):
    """
    `GT-D2G-path in paper`

    Edges are generated by another RNN
    """
    def __init__(self,
                 encoder_hidden_size: int,
                 attention_units: int,
                 max_out_node_size: int,
                 graph_rnn_num_layers: int,
                 graph_rnn_hidden_size: int,
                 edge_rnn_num_layers: int,
                 edge_rnn_hidden_size: int,
                 tau: int,
                 p_dropout: int):
        super(GPTGRNNDecoder, self).__init__()
        self.hidden_size = graph_rnn_hidden_size
        self.edge_hidden_size = edge_rnn_hidden_size
        self.max_out_node_size = max_out_node_size
        self.input_size = encoder_hidden_size + max_out_node_size - 1
        self.graph_rnn_num_layers = graph_rnn_num_layers
        self.edge_rnn_num_layers = edge_rnn_num_layers
        self.tau = tau
        # エンコーダの隠れ状態をグラフRNNの隠れ状態のサイズに投影する全結合層
        self.gcn_emb_projector = nn.Linear(encoder_hidden_size, graph_rnn_num_layers*self.hidden_size)
        #  グラフ構造を生成するためのGRU層
        self.graph_gru = nn.GRU(self.input_size, self.hidden_size, graph_rnn_num_layers,
                                batch_first=True, dropout=p_dropout)
        # エンコーダの出力に対してアテンションを計算するためのAttentionモジュール
        self.attention = Attention(self.hidden_size, attention_units)
        # need a linear layer to map dimension　グラフRNNの隠れ状態をエッジRNNの隠れ状態のサイズに投影する全結合層
        self.graph_edge_projector = nn.Linear(graph_rnn_num_layers*self.hidden_size,
                                              edge_rnn_num_layers*edge_rnn_hidden_size)
        # エッジの存在確率を生成するためのGRU層
        self.edge_gru = nn.GRU(1, edge_rnn_hidden_size, edge_rnn_num_layers,
                               batch_first=True)
        # エッジRNNの出力からエッジの存在確率を計算するための全結合層とシグモイド関数
        self.fout = nn.Sequential(
                nn.Linear(edge_rnn_hidden_size, edge_rnn_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(edge_rnn_hidden_size // 2, 1),
                nn.Sigmoid()
                )

    def forward(self, g: dgl.DGLGraph, h: th.Tensor, hg: th.Tensor) -> tuple:
        # h:  nnodes * hidden_s
        # hg: batch * hidden_s
        pointer_argmaxs = []  # store generated sequences 生成されたノードのインデックスを格納するリスト
        adj_vecs = []         # store generated adj_vecs　生成されたエッジの存在確率を表すリスト（テンソル）
        pseduo_seqs = []   # regard nodes of graph as a sequence　グラフのノードを疑似的なシーケンスとして扱うためのリスト、各グラフのノード特徴量を格納する
        nidx = 0
        # TODO: consider other pseudo sequence
        # バッチ内の各グラフのノード数を取得→ノード特徴量を格納
        for nnode in g.batch_num_nodes():
            pseduo_seqs.append(h[nidx:nidx+nnode, :])
            nidx += nnode
        # シーケンスの長さに基づいてマスクを生成　シーケンスの有効な部分を1，パディングされた部分を0とする
        # マスクはAttension機構、損失関数の計算において有効な部分だけを使用するために使われる
        mask = get_mask_from_sequence_lengths(g.batch_num_nodes(), g.batch_num_nodes().max())
        # パディングを追加 エンコーダの出力（デコーダの入力）を取得 グラフによってノード数が違うためノード数が最大のものに合わせる　これでバッチ処理ができる？
        encoder_out = pad_sequence(pseduo_seqs, batch_first=True)  # batch*max_nnode*hi
        batch_size = encoder_out.shape[0]
        decoder_input = encoder_out.new_zeros((batch_size, self.input_size)).unsqueeze(1)   # batch*1*input
        # グラフ全体の特徴量hgをデコーダの隠れ状態のサイズに変換し、ReLU活性化関数を適用
        decoder_hidden = self.gcn_emb_projector(hg).view(batch_size, self.graph_rnn_num_layers, self.hidden_size).relu()
        decoder_hidden = decoder_hidden.transpose(0, 1).contiguous()  # num_layer*batch*hidden
        # カバレッジベクトルの初期化
        cov_vec = encoder_out.new_zeros((batch_size, encoder_out.shape[1]))    # batch*max_nnode
        # カバレッジ損失の初期化　カバレッジベクトルの合計値
        cov_loss = cov_vec.sum()
    
        for i in range(self.max_out_node_size):
            # GRUで次の隠れ状態と出力を計算
            output, decoder_hidden = self.graph_gru(decoder_input, decoder_hidden)
            # output: batch * seq_len(=1) * hidden
            # decoder_hidden: num_layer * batch * hidden_size
            # アテンションスコアを計算
            att_scores = self.attention(encoder_out, output.squeeze(1), cov_vec, mask)
            # decode node
            # カバレッジ損失の計算
            cov_loss += th.minimum(att_scores, cov_vec).sum()
            cov_vec = cov_vec + att_scores
            # アテンションスコアに基づいて次のノードを選択するための確率分布を計算 tauは温度パラメータ hard=trueでone-hotベクトル表現になる
            att_argmax = F.gumbel_softmax((att_scores + 1e-12).log(),
                                          tau=self.tau, hard=True, dim=1)   # batch*max_nnode
            # 選択されたノードに対するエンコーダの出力を抽出　ノードの埋め込みを取得
            chosen_node_emb = th.matmul(encoder_out.transpose(1, 2),
                                        att_argmax.unsqueeze(-1)).squeeze(-1)  # batch*hidden
            pointer_argmaxs.append(att_argmax)
            # decode adj vector
            # エッジ生成
            theta = []
            # エッジRNNの隠れ状態の形に変換
            edge_hn = self.graph_edge_projector(decoder_hidden.transpose(0, 1).reshape(batch_size, -1)).relu()
            # edge_hn:batch*(edge_layer*edge_hidden)
            edge_hn = edge_hn.view(batch_size, self.edge_rnn_num_layers,
                                   self.edge_hidden_size).transpose(0, 1).contiguous()
            # edge_hn:edge_layer*batch*edge_hidden
            # エッジRNNの初期入力
            edge_input = encoder_out.new_zeros((batch_size, 1, 1))  # batch*seq_len(=1)*input(=1)
            # 選択した各ノードに対してエッジを予測
            for j in range(i):
                edge_output, edge_hn = self.edge_gru(edge_input, edge_hn)
                # edge_output: batch*seq_len(=1)*edge_hidden
                edge_input = self.fout(edge_output.squeeze(1))   # batch*1
                # 予測されたエッジの存在確率を追加
                theta.append(edge_input)
                edge_input = edge_input.unsqueeze(1)
            # 現在のノードとまだ生成されていないノードのエッジの存在確率を0にする
            for j in range(self.max_out_node_size-1-i):
                theta.append(encoder_out.new_zeros(batch_size, 1))
            # thetaリストの要素を連結することで現在のノードiと他の全てのノードとのエッジの存在確率が表現される
            theta = th.cat(theta, dim=1)  # batch*max_out
            adj_vecs.append(theta)
            # 選択されたノードの埋め込みと予測されたエッジの存在確率を連結することで新しいデコーダ入力を作成
            decoder_input = th.cat((chosen_node_emb, theta), dim=1)    # batch*input
            decoder_input = decoder_input.unsqueeze(1)    # batch*seq_len(=1)*input
        # pointer_argmaxs: batch*max_nnode*max_seq_len, cov_loss: scalar
        # encoder_out: batch*max_nnode*hidden, adj_vecs:batch*max_seq-1*max_seq_len
        # カバレッジ損失をバッチサイズで割って正規化
        cov_loss = cov_loss / batch_size
        return th.stack(pointer_argmaxs, dim=2), cov_loss, encoder_out, th.stack(adj_vecs, dim=2)


class GPTGRNNDecoderVariable(nn.Module):
    """
    GT-D2G-path-var version
    """
    def __init__(self,
                 encoder_hidden_size: int,
                 attention_units: int,
                 max_out_node_size: int,
                 graph_rnn_num_layers: int,
                 graph_rnn_hidden_size: int,
                 edge_rnn_num_layers: int,
                 edge_rnn_hidden_size: int,
                 tau: int,
                 p_dropout: int,
                 pos_emb_n_dim: int = 4):
        super(GPTGRNNDecoderVariable, self).__init__()
        self.hidden_size = graph_rnn_hidden_size
        self.edge_hidden_size = edge_rnn_hidden_size
        self.max_out_node_size = max_out_node_size
        self.input_size = encoder_hidden_size + max_out_node_size - 1 + pos_emb_n_dim
        self.graph_rnn_num_layers = graph_rnn_num_layers
        self.edge_rnn_num_layers = edge_rnn_num_layers
        self.tau = tau
        self.gcn_emb_projector = nn.Linear(encoder_hidden_size, graph_rnn_num_layers*self.hidden_size)
        self.special_tok_emb = nn.Embedding(2, encoder_hidden_size)   # 0: <sos>, 1: <eos>
        self.position_emb = nn.Embedding(max_out_node_size, pos_emb_n_dim)   # one extra space for <eos>
        self.graph_gru = nn.GRU(self.input_size, self.hidden_size, graph_rnn_num_layers,
                                batch_first=True, dropout=p_dropout)
        self.attention = Attention(self.hidden_size, attention_units)
        # need a linear layer to map dimension
        self.graph_edge_projector = nn.Linear(graph_rnn_num_layers*self.hidden_size,
                                              edge_rnn_num_layers*edge_rnn_hidden_size)
        self.edge_gru = nn.GRU(1, edge_rnn_hidden_size, edge_rnn_num_layers,
                               batch_first=True)
        self.fout = nn.Sequential(
                nn.Linear(edge_rnn_hidden_size, edge_rnn_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(edge_rnn_hidden_size // 2, 1),
                nn.Sigmoid()
                )
        # GRU weight init
        self.gcn_emb_projector.apply(weight_init)
        self.graph_gru.apply(weight_init)
        self.edge_gru.apply(weight_init)
        self.graph_edge_projector.apply(weight_init)

    def forward(self, g: dgl.DGLGraph, h: th.Tensor, hg: th.Tensor) -> tuple:
        # h:  nnodes * enc_hidden
        # hg: batch * enc_hidden
        pointer_argmaxs = []  # store generated sequences
        adj_vecs = []         # store generated adj_vecs
        pseduo_seqs = []   # regard nodes of graph as a sequence
        nidx = 0
        for nnode in g.batch_num_nodes():
            # pseduo_seqs.append(h[nidx:nidx+nnode, :])
            # pseudo_seqs with <eos> at the starting point
            cur_seq = th.cat((self.special_tok_emb(h.new_ones([1], dtype=th.long)), h[nidx:nidx+nnode, :]), 0)
            pseduo_seqs.append(cur_seq)
            nidx += nnode
        pseudo_seq_lens = g.batch_num_nodes() + 1   # size=(batch,)
        mask = get_mask_from_sequence_lengths(pseudo_seq_lens, pseudo_seq_lens.max())
        encoder_out = pad_sequence(pseduo_seqs, batch_first=True)  # batch*max_nnode+1*enc_hidden
        batch_size = encoder_out.shape[0]

        sos_seqs = self.special_tok_emb(h.new_zeros(batch_size, dtype=th.long))   # batch*enc_hid
        theta = encoder_out.new_zeros((batch_size, self.max_out_node_size-1))     # batch*max_out-1
        decoder_input = th.cat((sos_seqs, theta), dim=1).unsqueeze(1)             # batch*1*(enc_hid+max_out-1)
        decoder_hidden = self.gcn_emb_projector(hg).view(batch_size, self.graph_rnn_num_layers, self.hidden_size).relu()
        decoder_hidden = decoder_hidden.transpose(0, 1).contiguous()  # num_layer*batch*hidden
        cov_vec = encoder_out.new_zeros((batch_size, encoder_out.shape[1]))    # batch*max_nnode
        cov_loss = cov_vec.sum()
        att_scores = []
        for i in range(self.max_out_node_size):
            pos_emb = self.position_emb(h.new_tensor([i for _ in range(batch_size)],
                                                     dtype=th.long)).unsqueeze(1)   # batch*1*pos_emb
            decoder_input = th.cat((decoder_input, pos_emb), dim=2)   # batch*1*(enc_hid+max_out-1+pos_emb)
            output, decoder_hidden = self.graph_gru(decoder_input, decoder_hidden)
            # output: batch * seq_len(=1) * hidden
            # decoder_hidden: num_layer * batch * hidden_size
            att_score = self.attention(encoder_out, output.squeeze(1), cov_vec, mask)   # batch*max_nnode+1
            att_scores.append(att_score)
            # decode node
            att_argmax = F.gumbel_softmax((att_score + 1e-12).log(),
                                          tau=self.tau, hard=True, dim=1)   # batch*max_nnode+1
            chosen_node_emb = th.matmul(encoder_out.transpose(1, 2),
                                        att_argmax.unsqueeze(-1)).squeeze(-1)  # batch*hidden
            pointer_argmaxs.append(att_argmax)
            cov_loss += th.minimum(att_score, cov_vec).sum()
            cov_vec = cov_vec + att_score
            # decode adj vector
            theta = []
            edge_hn = self.graph_edge_projector(decoder_hidden.transpose(0, 1).reshape(batch_size, -1)).relu()
            # edge_hn:batch*(edge_layer*edge_hidden)
            edge_hn = edge_hn.view(batch_size, self.edge_rnn_num_layers,
                                   self.edge_hidden_size).transpose(0, 1).contiguous()
            # edge_hn:edge_layer*batch*edge_hidden
            edge_input = encoder_out.new_zeros((batch_size, 1, 1))  # batch*seq_len(=1)*input(=1)
            for j in range(i):
                edge_output, edge_hn = self.edge_gru(edge_input, edge_hn)
                # edge_output: batch*seq_len(=1)*edge_hidden
                edge_input = self.fout(edge_output.squeeze(1))   # batch*1
                theta.append(edge_input)
                edge_input = edge_input.unsqueeze(1)
            for j in range(self.max_out_node_size-1-i):
                theta.append(encoder_out.new_zeros(batch_size, 1))
            theta = th.cat(theta, dim=1)  # batch*max_out-1
            adj_vecs.append(theta)
            decoder_input = th.cat((chosen_node_emb, theta), dim=1)    # batch*enc_hid+max_out-1
            decoder_input = decoder_input.unsqueeze(1)    # batch*seq_len(=1)*enc_hid+max_out-1
        cov_loss = cov_loss / batch_size
        # pointer_argmaxs: batch*max_nnode+1*max_seq_len, cov_loss: (1,)
        # encoder_out: batch*max_nnode+1*hidden, adj_vecs:batch*max_seq-1*max_seq_len
        # att_scores: batch*max_nnode+1*max_seq_len
        return (th.stack(pointer_argmaxs, dim=2), cov_loss, encoder_out,
                th.stack(adj_vecs, dim=2), th.stack(att_scores, dim=2))


class GPTGRNNDecoderSVar(nn.Module):
    """
    GT-D2G-var in paper
    GPT-GRNN Decoder, Simple version(MLP for edges), variable length

    input_size eqauls to GCN emb_size +  adj_vec size
    hidden_size equals to GCN emb_size
    """
    def __init__(self,
                 encoder_hidden_size: int,
                 attention_units: int,
                 max_out_node_size: int,
                 graph_rnn_num_layers: int,
                 graph_rnn_hidden_size: int,
                 tau: int,
                 p_dropout: int = 0.0,
                 pos_emb_n_dim: int = 4):
        super(GPTGRNNDecoderSVar, self).__init__()
        self.hidden_size = graph_rnn_hidden_size
        self.max_out_node_size = max_out_node_size
        self.input_size = encoder_hidden_size + max_out_node_size - 1 + pos_emb_n_dim
        self.graph_rnn_num_layers = graph_rnn_num_layers
        self.tau = tau
        self.gcn_emb_projector = nn.Linear(encoder_hidden_size, graph_rnn_num_layers*self.hidden_size)
        self.special_tok_emb = nn.Embedding(2, encoder_hidden_size)   # 0: <sos>, 1: <eos>
        self.position_emb = nn.Embedding(max_out_node_size, pos_emb_n_dim)   # one extra space for <eos>
        self.graph_gru = nn.GRU(self.input_size, self.hidden_size, graph_rnn_num_layers,
                                batch_first=True, dropout=p_dropout)
        self.attention = Attention(self.hidden_size, attention_units)
        self.fout = nn.Sequential(
                nn.Linear(self.hidden_size, self.max_out_node_size - 1),
                nn.Sigmoid()
                )
        # GRU weight init
        self.gcn_emb_projector.apply(weight_init)
        self.graph_gru.apply(weight_init)

    def forward(self, g: dgl.DGLGraph, h: th.Tensor, hg: th.Tensor) -> tuple:
        # h:  nnodes * enc_hidden
        # hg: batch * enc_hidden
        pointer_argmaxs = []  # store generated sequences
        adj_vecs = []         # store generated adj_vecs
        pseduo_seqs = []   # regard nodes of graph as a sequence
        nidx = 0
        for nnode in g.batch_num_nodes():
            # pseduo_seqs.append(h[nidx:nidx+nnode, :])
            # pseudo_seqs with <eos> at the starting point
            cur_seq = th.cat((self.special_tok_emb(h.new_ones([1], dtype=th.long)), h[nidx:nidx+nnode, :]), 0)
            pseduo_seqs.append(cur_seq)
            nidx += nnode
        pseudo_seq_lens = g.batch_num_nodes() + 1   # size=(batch,)
        mask = get_mask_from_sequence_lengths(pseudo_seq_lens, pseudo_seq_lens.max())
        encoder_out = pad_sequence(pseduo_seqs, batch_first=True)  # batch*max_nnode+1*enc_hidden
        batch_size = encoder_out.shape[0]

        sos_seqs = self.special_tok_emb(h.new_zeros(batch_size, dtype=th.long))   # batch*enc_hid
        theta = encoder_out.new_zeros((batch_size, self.max_out_node_size-1))     # batch*max_out-1
        decoder_input = th.cat((sos_seqs, theta), dim=1).unsqueeze(1)             # batch*1*(enc_hid+max_out-1)
        decoder_hidden = self.gcn_emb_projector(hg).view(batch_size, self.graph_rnn_num_layers, self.hidden_size).tanh()
        decoder_hidden = decoder_hidden.transpose(0, 1).contiguous()  # num_layer*batch*hidden
        cov_vec = encoder_out.new_zeros((batch_size, encoder_out.shape[1]))    # batch*max_nnode
        cov_loss = cov_vec.sum()
        att_scores = []
        for i in range(self.max_out_node_size):
            pos_emb = self.position_emb(h.new_tensor([i for _ in range(batch_size)],
                                                     dtype=th.long)).unsqueeze(1)   # batch*1*pos_emb
            decoder_input = th.cat((decoder_input, pos_emb), dim=2)   # batch*1*(enc_hid+max_out-1+pos_emb)
            output, decoder_hidden = self.graph_gru(decoder_input, decoder_hidden)
            # output: batch * seq_len(=1) * hidden
            # decoder_hidden: num_layer * batch * hidden_size
            att_score = self.attention(encoder_out, output.squeeze(1), cov_vec, mask)   # batch*max_nnode+1
            att_scores.append(att_score)
            # decode node
            att_argmax = F.gumbel_softmax((att_score + 1e-12).log(),
                                          tau=self.tau, hard=True, dim=1)   # batch*max_nnode+1
            chosen_node_emb = th.matmul(encoder_out.transpose(1, 2),
                                        att_argmax.unsqueeze(-1)).squeeze(-1)  # batch*hidden
            pointer_argmaxs.append(att_argmax)
            cov_loss += th.minimum(att_score, cov_vec).sum()
            cov_vec = cov_vec + att_score
            # decode adj vector
            if i < 1:   # i==0, not generate
                theta = encoder_out.new_zeros((batch_size, self.max_out_node_size-1))  # batch*(max_out_seq_len-1)
            else:
                theta = self.fout(output.squeeze(1))       # batch*max_out-1, using soft weight for now
            adj_vecs.append(theta)
            decoder_input = th.cat((chosen_node_emb, theta), dim=1)    # batch*enc_hid+max_out-1
            decoder_input = decoder_input.unsqueeze(1)    # batch*seq_len(=1)*enc_hid+max_out-1
        cov_loss = cov_loss / batch_size
        # pointer_argmaxs: batch*max_nnode+1*max_seq_len, cov_loss: (1,)
        # encoder_out: batch*max_nnode+1*hidden, adj_vecs:batch*max_seq-1*max_seq_len
        # att_scores: batch*max_nnode+1*max_seq_len
        return (th.stack(pointer_argmaxs, dim=2), cov_loss, encoder_out,
                th.stack(adj_vecs, dim=2), th.stack(att_scores, dim=2))
