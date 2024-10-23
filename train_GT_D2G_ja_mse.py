"""
Training scripts for GPT-GRNN Completed Version
Author: 
Create Date: Dec 16, 2020
"""

import time
import random

import numpy as np
import torch as th
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sacred
from sacred.observers import FileStorageObserver
import japanize_matplotlib

from utils import convert_adj_vec_to_matrix
from model.data_loader_mse import prepare_ingredients, collate_fn
from model.GPT_GRNN import GCNEncoder, GPTGRNNDecoder, GraphClassifier
import dgl
import os


# Sacred Setup
ex = sacred.Experiment('train_GT-D2G-ja')
ex.observers.append(FileStorageObserver("logs/GT-D2G-ja-pptx"))

def count_ner(pointer_argmax, nodes_text, ner_li):
    nodes = th.argmax(pointer_argmax, dim=0)  # 1の値のインデックスを取り出す

    nodes_text_li = list(nodes_text)

    count = 0
    for node in nodes:
        if nodes_text_li[int(node)] in ner_li:
            count += 1

    return th.tensor(count)
def visualize_graph_with_text(seed, pointer_argmax, adj_matrix, nodes_text, i_epoch, i_batch, docid, filename, mode):
    nodes = th.argmax(pointer_argmax, dim=0)  # 1の値のインデックスを取り出す
    G = nx.Graph()

    nodes_text_li = list(nodes_text)
    choice_nodes = []

    for node in nodes:
        choice_nodes.append(nodes_text_li[int(node)])
        G.add_node(nodes_text_li[int(node)])

    # エッジの存在確率が最大の部分だけ接続
    for i, probability in enumerate(adj_matrix):
        filterd_pro_list = [value for value in probability if value != 1.0]

        max_pro = max(filterd_pro_list)
        probability = probability.tolist()  # tensor -> list
        max_index = probability.index(max_pro)
        if max_pro.item() > 0.6:
            G.add_edge(choice_nodes[int(i)], choice_nodes[int(max_index)])

    # サイズ
    plt.figure(figsize=(15, 15))
    # グラフの描画
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, font_size=30, with_labels=True, alpha=0.5, font_family='IPAexGothic')

    filename = filename.split('/')[-1]
    filename = f'imgs/japanese-pptx/{seed}/{mode}/{i_epoch}epoch_{i_batch}batch_{docid}_{filename}.png'
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # 保存
    plt.axis("off")
    # print(f'i_epoch,i_batch = {i_epoch,i_batch}')
    plt.savefig(filename)
    plt.close()

@ex.config
def my_config():
    motivation = ""
    opt = {
           'gpu': False,
           'seed': 27,
           'corpus_type': 'ld',    # 'yelp'|'dblp'|'nyt'|'ld'|'pptx'
           'processed_pickle_path': '',
           'checkpoint_dir': '',
           'n_labels': {
               'nyt': 5,
               'yelp': 5,
               'ld': 9,
               'yelp-3-class': 3,
               'dblp': 6,
               'pptx': 9
               },
           'epoch': 250,
           'epoch_warmup': 10,
           'early_stop_flag': False,
           'patience': 100,
           'batch_size': 128,
           'lr': 3e-4,
           'lr_scheduler_cosine_T_max': 64,
           'optimizer_weight_decay': 0.0,
           'lambda_cov_loss': 0.1,
           'shrinkage_lambda_cov_per_epoch': 50,
           'shrinkage_rate_lambda_cov': 0.25,
           'clip_grad_norm': 5.0,
           'gptrnn_decoder_dropout': 0.3,
           'gcn_encoder_hidden_size': 128,
           'gcn_encoder_pooling': 'mean',
           'graph_rnn_num_layers': 2,
           'graph_rnn_hidden_size': 128,
           'edge_rnn_num_layers': 2,
           'edge_rnn_hidden_size': 16,
           'GPT_attention_unit': 10,
           'max_out_node_size': 20,
           'gumbel_tau': 3,
           'gumbel_tau_min': 0.5,
           'gumbel_tau_decay': 0.995,
           'gpt_grnn_variant': 'path',
           'gcn_classifier_hidden_size': 64,
           'pretrain_emb_name': 'model.vec',
           'pretrain_emb_cache': None,
           'pretrain_emb_max_vectors': 160000,
           'yelp_senti_feat': False,
           'pretrain_emb_dropout': 0.0,
           'regular_loss_decay': 0.993,
           'resume_checkpoint_path': '',
          }


@ex.automain
def train_model(opt, _run, _log):
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    th.manual_seed(opt['seed'])
    _log.info("The random seed has been set to %d globally" % (opt['seed']))

    # Sanity check
    if not opt['corpus_type'] or not opt['processed_pickle_path'] or not opt['checkpoint_dir']:
        _log.error('missing essential input arguments')
        exit(-1)
    n_labels = opt['n_labels'][opt['corpus_type']]
    lambda_cov_loss = opt['lambda_cov_loss']

    # Load corpus
    batch_size = opt['batch_size']
    pickle_path = opt['processed_pickle_path']
    _log.info('[%s] Start loading %s corpus from %s' % (time.ctime(), opt['corpus_type'], pickle_path))
    train_set, val_set, test_set, vocab = prepare_ingredients(pickle_path, corpus_type=opt['corpus_type'],
                                                              pretrain_name=opt['pretrain_emb_name'],
                                                              emb_cache=opt['pretrain_emb_cache'],
                                                              max_vectors=opt['pretrain_emb_max_vectors'],
                                                              yelp_senti_feature=opt['yelp_senti_feat'])
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_iter = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    _log.info('[%s] Load train, val, test sets Done, len=%d,%d,%d' % (time.ctime(),
              len(train_set), len(val_set), len(test_set)))

    # Build models
    pretrained_emb = vocab.vectors
    gcn_encoder = GCNEncoder(pretrained_emb, pretrained_emb.shape[1]+3, opt['gcn_encoder_hidden_size'],
                             opt['gcn_encoder_pooling'], opt['yelp_senti_feat'],
                             opt['pretrain_emb_dropout'])
    gptrnn_decoder = GPTGRNNDecoder(opt['gcn_encoder_hidden_size'], opt['GPT_attention_unit'],
                                    opt['max_out_node_size'], opt['graph_rnn_num_layers'],
                                    opt['graph_rnn_hidden_size'], opt['edge_rnn_num_layers'],
                                    opt['edge_rnn_hidden_size'],
                                    opt['gumbel_tau'], opt['gptrnn_decoder_dropout']
                                    )
    gcn_classifier = GraphClassifier(opt['gcn_encoder_hidden_size'], opt['gcn_classifier_hidden_size'],
                                     n_labels)
    class_criterion = nn.CrossEntropyLoss()
    class_mse = nn.MSELoss()
    parameters = list(gcn_encoder.parameters()) + list(gptrnn_decoder.parameters()) \
        + list(gcn_classifier.parameters())
    optimizer = th.optim.Adam(parameters, opt['lr'], weight_decay=opt['optimizer_weight_decay'])
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt['lr_scheduler_cosine_T_max'])
    if opt['resume_checkpoint_path']:
        checkpoint = th.load(opt['resume_checkpoint_path'])
        gcn_encoder.load_state_dict(checkpoint['gcn_encoder'])
        gptrnn_decoder.load_state_dict(checkpoint['gptrnn_decoder'])
        gcn_classifier.load_state_dict(checkpoint['gcn_classifier'])
        _log.info('[%s] Load resume_checkpoint_path from %s' % (time.ctime(), opt['resume_checkpoint_path']))
    if opt['gpu']:
        gcn_encoder = gcn_encoder.cuda()
        gptrnn_decoder = gptrnn_decoder.cuda()
        gcn_classifier = gcn_classifier.cuda()
        class_criterion = class_criterion.cuda()

    # Start Epochs
    max_acc = 0.0
    for i_epoch in range(opt['epoch']):
        # Start Training
        gcn_encoder.train()
        gptrnn_decoder.train()
        gcn_classifier.train()
        train_loss = []
        train_class_loss = []
        train_cov_loss = []
        all_train_pred = []
        all_train_gold = []
        for i_batch, batch in enumerate(train_iter):
            optimizer.zero_grad()
            batched_graph, nid_mappings, labels, docids, nodes, filenames, ner = batch
            batch_size = labels.shape[0]
            if opt['gpu']:
                batched_graph = batched_graph.to('cuda:0')
                labels = labels.cuda()
            h, hg = gcn_encoder(batched_graph)
            pointer_argmaxs, cov_loss, encoder_out, adj_vecs = gptrnn_decoder(batched_graph, h, hg)
            adj_matrix = convert_adj_vec_to_matrix(adj_vecs, add_self_loop=True)
            generated_nodes_emb = th.matmul(pointer_argmaxs.transpose(1, 2), encoder_out)  # batch*seq_l*hid
            pred = gcn_classifier(generated_nodes_emb, adj_matrix)
            # 専門用語の出現回数をカウント
            ner_count_all = th.tensor([])
            for i in range(len(pointer_argmaxs)):
                ner_count = count_ner(pointer_argmaxs[i], nodes[i], ner[i])
                ner_count_all = th.cat([ner_count_all, th.tensor([ner_count])])

            if i_epoch == 0 or i_epoch == 5 or i_epoch == 10 or i_epoch == 100 or i_epoch == 240 or i_epoch == 250 or i_epoch == 260 or i_epoch == 300:
                for i in range(len(pointer_argmaxs)):


                    if i < len(pointer_argmaxs) and i < len(adj_matrix) and i < len(nodes) and i < len(docids):
                        visualize_graph_with_text(opt['seed'], pointer_argmaxs[i], adj_matrix[i], nodes[i],
                                              i_epoch, i_batch, docids[i], filenames[i], 'train')
                    else:
                        _log.info('Index {%d} is out!' % i)
            # 実際の専門用語の出現回数
            y_ner_count_all = list(map(len, ner))
            y_ner_count_all = th.LongTensor(y_ner_count_all)
            # 専門用語の回数を損失として扱う
            class_mse_loss = class_mse(ner_count_all, y_ner_count_all)
            class_loss = class_criterion(pred, labels)
            loss = class_loss + class_mse_loss + lambda_cov_loss * cov_loss
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=opt['clip_grad_norm'], norm_type=2)
            optimizer.step()
            train_loss.append(loss.item())
            train_class_loss.append(class_loss.item())
            train_cov_loss.append(cov_loss.item())
            all_train_gold.extend(labels.detach().tolist())
            all_train_pred.extend(th.argmax(pred, dim=1).detach().tolist())
        avg_loss = sum(train_loss)/len(train_loss)
        train_acc = (th.LongTensor(all_train_gold) == th.LongTensor(all_train_pred)).sum() / len(all_train_pred)
        _run.log_scalar("train.loss", avg_loss, i_epoch)
        _run.log_scalar("train.class_loss", sum(train_class_loss)/len(train_class_loss), i_epoch)
        _run.log_scalar("train.cov_loss", sum(train_cov_loss)/len(train_cov_loss), i_epoch)
        _run.log_scalar("train.acc", train_acc * 100, i_epoch)
        _log.info('[%s] epoch#%d train Done, avg loss=%.5f, train_acc=%.2f' % (time.ctime(), i_epoch, avg_loss, train_acc*100))
        # Start Validating
        gcn_encoder.eval()
        gptrnn_decoder.eval()
        gcn_classifier.eval()
        val_loss = []
        val_class_loss = []
        val_cov_loss = []
        all_pred = []
        all_gold = []
        with th.no_grad():
            for i_batch, batch in enumerate(val_iter):
                batched_graph, nid_mappings, labels, docids, nodes, filenames,ner  = batch
                batch_size = labels.shape[0]
                if opt['gpu']:
                    batched_graph = batched_graph.to('cuda:0')
                    labels = labels.cuda()
                h, hg = gcn_encoder(batched_graph)
                pointer_argmaxs, cov_loss, encoder_out, adj_vecs = gptrnn_decoder(batched_graph, h, hg)
                adj_matrix = convert_adj_vec_to_matrix(adj_vecs, add_self_loop=True)
                generated_nodes_emb = th.matmul(pointer_argmaxs.transpose(1, 2), encoder_out)  # batch*seq_l*hid
                pred = gcn_classifier(generated_nodes_emb, adj_matrix)
                # 専門用語の出現回数をカウント
                ner_count_all = th.tensor([])
                for i in range(len(pointer_argmaxs)):
                    ner_count = count_ner(pointer_argmaxs[i], nodes[i], ner[i])
                    ner_count_all = th.cat([ner_count_all, th.tensor([ner_count])])
                if i_epoch == 0 or i_epoch == 5 or i_epoch == 200 or i_epoch == 250:
                    for i in range(len(pointer_argmaxs)):
                        if i < len(pointer_argmaxs) and i < len(adj_matrix) and i < len(nodes) and i < len(docids):
                            visualize_graph_with_text(opt['seed'], pointer_argmaxs[i], adj_matrix[i], nodes[i],
                                                      i_epoch, i_batch, docids[i], filenames[i], 'val')
                        else:
                            _log.info('Index {%d} is out!' % i)

                # 実際の専門用語の出現回数
                y_ner_count_all = list(map(len, ner))
                y_ner_count_all = th.LongTensor(y_ner_count_all)
                # 専門用語の回数を損失として扱う
                class_mse_loss = class_mse(ner_count_all, y_ner_count_all)
                class_loss = class_criterion(pred, labels)
                loss = class_loss + class_mse_loss + lambda_cov_loss * cov_loss
                val_loss.append(loss.item())
                val_class_loss.append(class_loss.item())
                val_cov_loss.append(cov_loss.item())
                all_gold.extend(labels.detach().tolist())
                all_pred.extend(th.argmax(pred, dim=1).detach().tolist())
        avg_loss = sum(val_loss) / len(val_loss)
        acc = (th.LongTensor(all_gold) == th.LongTensor(all_pred)).sum() / len(all_pred)
        _run.log_scalar("eval.loss", avg_loss, i_epoch)
        _run.log_scalar("eval.class_loss", sum(val_class_loss)/len(val_class_loss), i_epoch)
        _run.log_scalar("eval.cov_loss", sum(val_cov_loss)/len(val_cov_loss), i_epoch)
        _run.log_scalar("eval.acc", acc*100, i_epoch)
        _log.info('[%s] epoch#%d validation Done, avg loss=%.5f, acc=%.2f' % (time.ctime(), i_epoch,
                                                                              avg_loss, acc * 100))
        if i_epoch > opt['epoch_warmup']:
            if acc > max_acc:
                max_acc = acc
                save_path = '%s/exp%s_%s.best.ckpt' % (opt['checkpoint_dir'], _run._id, opt['corpus_type'])
                _log.info('Achieve best acc, store model into %s' % (save_path))
                th.save({'gcn_encoder': gcn_encoder.state_dict(),
                         'gptrnn_decoder': gptrnn_decoder.state_dict(),
                         'gcn_classifier': gcn_classifier.state_dict()
                         }, save_path)
                patience = 0
            else:
                patience += 1
            # early stop
            if opt['early_stop_flag'] and patience > opt['patience']:
                _log.info('Achieve best acc=%.2f, early stop at epoch #%d' % (max_acc*100, i_epoch))
                exit(0)
        # scheduler
        scheduler.step()
        # Anneal gumbel-softmax tau
        gptrnn_decoder.tau = max(gptrnn_decoder.tau * opt['gumbel_tau_decay'], opt['gumbel_tau_min'])
        # Anneal regularization loss
        lambda_cov_loss *= opt['regular_loss_decay']   # can open or close
