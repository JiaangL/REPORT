import argparse
import networkx as nx
import os 
import re
import json
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from reader.vocab_reader import Vocabulary
from preprocess_ind_data import filter_repeat_path
import multiprocessing as mp


def build_graph(raw_train):
        biG = nx.MultiDiGraph()
        with open(raw_train, 'r') as f_train:
            for line in f_train.readlines():
                tokens = re.split(r'\t|\s', line.strip())
                head = tokens[0]
                relation = tokens[1]
                tail = tokens[2]
                biG.add_edge(head, tail, **{'r_type': relation})
                biG.add_edge(tail, head, **{'r_type': 'inv_' + relation})
        return biG


def convert_neg_sample(examples, n):     
    relation = [examples[i][0] for i in range(n)]
    head = [examples[i][1]for i in range(n)]
    tail = [examples[i][2]for i in range(n)]
    paths = [examples[i][3]for i in range(n)]
    num_path = [examples[i][4]for i in range(n)]
    path_mask = [examples[i][5]for i in range(n)]
    overall_mask = [examples[i][6]for i in range(n)]
    neg_triplet = [relation, head, tail, paths, num_path, path_mask, overall_mask]
    
    return neg_triplet


def find_path(obj):
    
    node_list = list(biG.nodes) #负样本采样的范围
    n = len(node_list)
    neg_triplets = {}
    
    relation = obj['relation']
    head = obj['head']
    tail = obj['tail']
    positive_id = obj['positive_id']
    relation_id = vocab.convert_tokens_to_ids([relation])[0]

    neg_heads, neg_tails = [head], [tail]
    #randomly sample neg_heads and neg_tails
    while len(neg_heads) < n_sample:
        neg_head = node_list[np.random.choice(n)]
        if neg_head != tail and neg_head not in neg_heads:
            flag = 1 #filter positive candidates
            if biG.has_edge(neg_head, tail):
                for i in biG[neg_head][tail]:
                    if biG[neg_head][tail][i]['r_type'] == relation:
                        flag = 0
            if flag == 1:
                neg_heads.append(neg_head)

    while len(neg_tails) < n_sample:
        neg_tail = node_list[np.random.choice(n)]
        if neg_tail != head and neg_tail not in neg_tails:
            flag = 1 #filter positive candidates
            if biG.has_edge(head, neg_tail):
                for i in biG[head][neg_tail]:
                    if biG[head][neg_tail][i]['r_type'] == relation:
                        flag = 0
            if flag == 1:
                neg_tails.append(neg_tail)

    neg_tails_path = {n_tail: [] for n_tail in neg_tails}
    neg_tails_pathmask = {n_tail: [] for n_tail in neg_tails}
    neg_heads_path = {n_head: [] for n_head in neg_heads}
    neg_heads_pathmask = {n_head: [] for n_head in neg_heads}
    
    for ntail in neg_tails:
        paths = nx.all_simple_edge_paths(biG, head, ntail, 4)
        for path in paths:
            r_path = [biG[edge[0]][edge[1]][edge[2]]
                ['r_type'] for edge in path]
            if len(r_path) == 1 and r_path[0] == relation:
                continue                  
            neg_tails_path[ntail].append(r_path)

    for nhead in neg_heads:
        paths = nx.all_simple_edge_paths(biG, nhead, tail, 4)
        for path in paths:
            r_path = [biG[edge[0]][edge[1]][edge[2]]
                ['r_type'] for edge in path]
            if len(r_path) == 1 and r_path[0] == relation:
                continue
            neg_heads_path[nhead].append(r_path)

    #filter repeated paths & add path mask
    for neg_tail in neg_tails:
        neg_tails_path[neg_tail] = filter_repeat_path(neg_tails_path[neg_tail])
        tmp = []
        for path in neg_tails_path[neg_tail]:
            tmp_mask = [1 for i in range(len(path))]
            while len(path) < args.max_path_len:
                path.append('[PAD]')
                tmp_mask.append(0)
            path.insert(0, '[CLS]')
            tmp_mask.insert(0, 1)
            neg_tails_pathmask[neg_tail].append(tmp_mask)
            tmp.append(vocab.convert_tokens_to_ids(path))
        
        neg_tails_path[neg_tail] = tmp

    for neg_head in neg_heads:
        neg_heads_path[neg_head] = filter_repeat_path(neg_heads_path[neg_head])
        tmp = []
        for path in neg_heads_path[neg_head]:
            tmp_mask = [1 for i in range(len(path))]
            while len(path) < args.max_path_len:
                path.append('[PAD]')
                tmp_mask.append(0)
            path.insert(0, '[CLS]')
            tmp_mask.insert(0, 1)
            neg_heads_pathmask[neg_head].append(tmp_mask)
            tmp.append(vocab.convert_tokens_to_ids(path))
        
        neg_heads_path[neg_head] = tmp

    # convert data to list of examples
    neg_head_examples = [[relation_id, 
            vocab.convert_tokens_to_ids(r_context[neg_heads[i]]), 
            vocab.convert_tokens_to_ids(r_context[tail]), 
            neg_heads_path[neg_heads[i]],
            len(neg_heads_path[neg_heads[i]]), 
            neg_heads_pathmask[neg_heads[i]]] 
            for i in range(n_sample)]
    neg_tail_examples = [[relation_id, 
            vocab.convert_tokens_to_ids(r_context[head]), 
            vocab.convert_tokens_to_ids(r_context[neg_tails[i]]), 
            neg_tails_path[neg_tails[i]],
            len(neg_tails_path[neg_tails[i]]), 
            neg_tails_pathmask[neg_tails[i]]] 
            for i in range(n_sample)]
    
    # add overall mask
    for example in neg_head_examples:
        overall_mask = [1] * (example[4] + 4) + [0] * (args.max_num_path - example[4])
        # 3 for [MASK]\head\tail
        example.append(overall_mask)
    for example in neg_tail_examples:
        overall_mask = [1] * (example[4] + 4) + [0] * (args.max_num_path - example[4])
        example.append(overall_mask)
    
    if path_mode == 'eval':
        neg_triplet = {}
        neg_triplet['neg_head_examples'] = convert_neg_sample(neg_head_examples, n_sample)
        neg_triplet['neg_tail_examples'] = convert_neg_sample(neg_tail_examples, n_sample)
        #neg_triplets[positive_id] = neg_triplet
    else:
        #训练时，随机决定替换h or t
        tmp = []
        for i in range(n_sample):
            if np.random.uniform() < 0.5:
                tmp.append(neg_head_examples[i])
            else:
                tmp.append(neg_tail_examples[i])
                #neg_triplets[positive_id] = convert_neg_sample(neg_tail_examples, num_sample)
        #neg_triplets[positive_id] = convert_neg_sample(tmp, num_sample)
        neg_triplet = convert_neg_sample(tmp, n_sample)

    return neg_triplet, positive_id


def get_neg_sampling_replacing_head_tail(raw_train, vocab_path, 
                relation_context, raw_predict, num_sample=50, mode='eval'):

    global biG, vocab, n_sample, r_context, path_mode
    path_mode = mode
    biG = build_graph(raw_train)
    vocab = Vocabulary(vocab_file=vocab_path)
    n_sample = num_sample
    with open(relation_context, 'r') as f_dict:
        r_context = json.load(f_dict)
    node_list = list(biG.nodes) #负样本采样的范围
    n = len(node_list)
    neg_triplets = {}
    n_line = 0
    
    positive_list = []
    with open(raw_predict, 'r') as fr:
        for line in fr.readlines():
            tmp = {}
            obj = json.loads(line.strip())
            tmp['relation'] = obj['relation']
            tmp['head'] = obj['head']
            tmp['tail'] = obj['tail']
            tmp['positive_id'] = obj['positive_id']
            positive_list.append(tmp)
        
    with mp.Pool(processes=None) as pool:
        for (neg_triplet, pos_id) in tqdm(pool.imap_unordered(find_path, positive_list, 
                chunksize=16), total=len(positive_list), desc=' Negative Sampling...'):
            neg_triplets[pos_id] = neg_triplet
    
    return neg_triplets
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        default='fb237_v1',
        choices=[
            'fb237_v1', 'fb237_v2', 'fb237_v3', 'fb237_v4', 'nell_v1', 'nell_v2', 
            'nell_v3', 'nell_v4', 'WN18RR_v1', 'WN18RR_v2', 'WN18RR_v3', 'WN18RR_v4',
            'fb237_v1_ind', 'fb237_v2_ind', 'fb237_v3_ind', 'fb237_v4_ind', 'nell_v1_ind', 'nell_v2_ind', 
            'nell_v3_ind', 'nell_v4_ind', 'WN18RR_v1_ind', 'WN18RR_v2_ind', 'WN18RR_v3_ind', 'WN18RR_v4_ind'
        ])
    parser.add_argument('--max_path_len', type=int, default=4)
    parser.add_argument('--max_num_path', type=int, default=900)
    args = parser.parse_args()
    data_dir = os.path.join(os.getcwd(), 'neg_samples')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    task_dir = os.path.join(data_dir, args.task) #处理后的负样本保存的路径
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    raw_train = os.path.join(os.getcwd(), 'data_raw', args.task, 'train.txt')
    train_file = os.path.join(os.getcwd(), 'data_preprocessed', args.task, 'train.json')
    test_file= os.path.join(os.getcwd(), 'data_preprocessed', args.task, 'test.json')
    valid_file = os.path.join(os.getcwd(), 'data_preprocessed', args.task, 'valid.json')
    
    ent_r_nbr = os.path.join('data_preprocessed', args.task, 'ent_r_nbr.json')
    """ test_r_nbr = os.path.join('data_preprocessed', args.task, 'test_r_nbr.json')
    valid_r_nbr = os.path.join('data_preprocessed', args.task, 'valid_r_nbr.json') """
    neg_save_path_valid = os.path.join(task_dir, 'neg_sample_valid.npy')
    neg_save_path_test = os.path.join(task_dir, 'neg_sample_test.npy')
    neg_save_path_train = os.path.join(task_dir, 'neg_sample_train.npy')
    if args.task[-3:] == 'ind':
        #在test时应该取和train graph中相同的vocabulary
        vocab_path = os.path.join('data_preprocessed', args.task[:-4], 'vocab_rel.txt')
    else:
        vocab_path = os.path.join('data_preprocessed', args.task, 'vocab_rel.txt')

    #在train-graph上，算指标用valid set里的query
    if args.task[-3:] == 'ind':
        print("It's a ind test graph set!")
        neg_triplets = get_neg_sampling_replacing_head_tail(
                raw_train=raw_train, vocab_path=vocab_path, 
                relation_context=ent_r_nbr, 
                raw_predict=test_file)
        np.save(neg_save_path_test, neg_triplets)
    else:
        #训练用的负样本
        neg_triplets = get_neg_sampling_replacing_head_tail(
                raw_train=raw_train, vocab_path=vocab_path, 
                relation_context=ent_r_nbr, 
                raw_predict=train_file, num_sample=11, mode='train')
        np.save(neg_save_path_train, neg_triplets)
        #测试用的负样本
        neg_triplets = get_neg_sampling_replacing_head_tail(
                raw_train=raw_train, vocab_path=vocab_path, 
                relation_context=ent_r_nbr, 
                raw_predict=valid_file)
        np.save(neg_save_path_valid, np.array(neg_triplets, dtype=object))
    print('DONE')