#preprosess inductive data
import collections
import json
import re
import os
import argparse
import logging

import networkx as nx
import multiprocessing as mp
from rich.progress import track
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.DEBUG,
    filename='preprocess_ind_data.log',
    filemode='a')


def filter_repeat_path(relation_path):
    filtered_path = []
    for path in relation_path:
        flag = 1
        for non_repeat_path in filtered_path:
            if path == non_repeat_path:
                flag = -1
                break
        if flag == 1:
            filtered_path.append(path)
    return filtered_path


class DataProcessor(object):

    def __init__(self, raw_train, raw_test, raw_valid, train, test, valid,
            ent_voc, rel_voc, ent_r_nbr, neg_test, neg_valid):
        self.raw_train = raw_train
        self.raw_valid = raw_valid
        self.raw_test = raw_test
   
        self.train = train
        self.valid = valid
        self.test = test
        self.ent_voc = ent_voc
        self.rel_voc = rel_voc
        self.ent_r_nbr = ent_r_nbr

        self.neg_test, self.neg_valid = neg_test, neg_valid

    def build_graph(self):
        self.biG = nx.MultiDiGraph()
        total_links = 0
        with open(self.raw_train, 'r') as f_train:
            for line in f_train.readlines():
                total_links = total_links + 1
                tokens = re.split(r'\t|\s', line.strip())
                head = tokens[0]
                relation = tokens[1]
                tail = tokens[2]
                self.biG.add_edge(head, tail, **{'r_type': relation})
                self.biG.add_edge(tail, head, **{'r_type': 'inv_' + relation})
        f_train.close()
        print("number of links in train is:{links}".format(links=total_links))
        
    def find_path(self, idx):
        
        obj = self.positive_obj[idx]
        head = obj['head']
        tail = obj['tail']
        relation = obj['relation']
        relation_path = []
        """ 将路径长度从1增长到4，直到找到路径 """
        for path in nx.all_simple_edge_paths(self.biG, head, tail, 4):
            r_path = [self.biG[edge[0]][edge[1]][edge[2]]
                ['r_type'] for edge in path]
            if len(r_path) == 1 and r_path[0] == relation:
                continue
            relation_path.append(r_path)
        relation_path = filter_repeat_path(relation_path)

        return relation_path, idx

    def build_query_data(self, query_data, query_output):
        #build input data
        #Query data is what we need to reason. 
        #Query output is a file with h\r\t\r-path for each triplet, 
        #where r is to be predicted with h\t\r-path.
        total_links = 0
        n_zero_path_ent_pairs = 0
        max_num_path = 0
        self.positive_obj = dict()
        with open(query_data, 'r') as f_r:
            for line in f_r.readlines():
                total_links = total_links + 1
                tokens = re.split(r'\t|\s', line.strip())
                head = tokens[0]
                relation = tokens[1]
                tail = tokens[2]

                new_obj = collections.OrderedDict()
                new_obj['relation'] = relation
                new_obj['head'] = head
                new_obj['tail'] = tail
                self.positive_obj[total_links] = new_obj

        with open(query_output, 'w') as fw:
            with mp.Pool(processes=None) as pool:
                inputs = range(1, total_links + 1)
                for (relation_path, pos_id) in tqdm(pool.imap(self.find_path, inputs, chunksize=16), 
                        total=total_links, desc=query_data + ' Extracting Path...'):
                    new_obj = self.positive_obj[pos_id]
                    new_obj['positive_id'] = pos_id
                    new_obj['path'] = relation_path
                    new_obj['num_path'] = len(relation_path)
                    fw.write(json.dumps(new_obj) + "\n")
                    max_num_path = max(new_obj['num_path'], max_num_path)
                    if len(relation_path) == 0:
                        n_zero_path_ent_pairs += 1
            
        print("number of links in {file} is:{links}. number of zero path entity pairs is {zeros}. max number of path is {max_path}"
            .format(file=query_data, links=total_links, zeros=n_zero_path_ent_pairs, max_path=max_num_path))    

    def build_query_data_old(self, query_data, query_output):
        #build input data
        #Query data is what we need to reason. 
        #Query output is a file with h\r\t\r-path for each triplet, 
        #where r is to be predicted with h\t\r-path.
        total_links = 0
        n_zero_path_ent_pairs = 0
        max_num_path = 0
        all_obj = dict()
        with open(query_data, 'r') as f_valid:
            with open(query_output, 'w') as fw:
                for line in tqdm(f_valid.readlines(), desc=query_data + ' Processing...'):
                    total_links = total_links + 1
                    tokens = re.split(r'\t|\s', line.strip())
                    head = tokens[0]
                    relation = tokens[1]
                    tail = tokens[2]
                    relation_path = []
                    
                    """ 将路径长度从1增长到4，直到找到路径 """
                    for path in nx.all_simple_edge_paths(self.biG, head, tail, 4):
                        r_path = [self.biG[edge[0]][edge[1]][edge[2]]
                            ['r_type'] for edge in path]
                        if len(r_path) == 1 and r_path[0] == relation:
                            continue
                        relation_path.append(r_path)
                    if len(relation_path) == 0:
                        n_zero_path_ent_pairs += 1
                    #4-hop之内找不到，就当作没有路径处理

                    new_obj = collections.OrderedDict()
                    new_obj['relation'] = relation
                    new_obj['head'] = head
                    new_obj['tail'] = tail
                    new_obj['positive_id'] = total_links
                    new_obj['path'] = filter_repeat_path(relation_path)
                    new_obj['num_path'] = len(new_obj['path'])
                    
                    fw.write(json.dumps(new_obj) + "\n")
                    max_num_path = max(new_obj['num_path'], max_num_path)
                    del relation_path
        print("number of links in {file} is:{links}. number of zero path entity pairs is {zeros}. max number of path is {max_path}"
            .format(file=query_data, links=total_links, zeros=n_zero_path_ent_pairs, max_path=max_num_path))    

    def build_relation_context(self, query_data, relation_context_output):
        ent_context = dict()
        with open(query_data, 'r') as fr:
            with open(relation_context_output, 'w') as fw:
                for line in fr.readlines():
                    tokens = re.split(r'\t|\s', line.strip())
                    head = tokens[0]
                    tail = tokens[2]

                    for ent in [head, tail]:
                        if ent not in ent_context.keys():
                            r_nbr = set()
                            #对于valid，用biG
                            for _, v in self.biG.out_edges(ent):
                                for n_edge in self.biG[ent][v]:
                                    r_type = self.biG[ent][v][n_edge]['r_type']
                                    r_nbr.add(r_type) #只包含出边或入边

                                    """ if r_type.startswith('inv_'):
                                        inv_r = r_type[4::]
                                    else:
                                        inv_r = 'inv_' + r_type
                                    r_nbr.add(inv_r)
                                    #nbr包含in and out的所有边和相应反关系 """
                                
                            ent_context[ent] = list(r_nbr)
                fw.write(json.dumps(ent_context) + '\n')
        print('finish building relation context')  

    def write_vocab(self, ent_list, rel_list):
        fout_ent_voc = open(self.ent_voc, "w")
        fout_rel_voc = open(self.rel_voc, "w")
        fout_rel_voc.write("[PAD]" + "\n")
        fout_rel_voc.write("[MASK]" + "\n")
        fout_rel_voc.write("[CLS]" + "\n")
        for ent in sorted(ent_list.keys()):
            fout_ent_voc.write(ent + '\n')
        for rel in sorted(rel_list.keys()):
            fout_rel_voc.write(rel + '\n')
        fout_ent_voc.close()
        fout_rel_voc.close()      

    def get_unique_roles_values(self):
        rel_dict = dict()
        ent_dict = dict()
        file_list = [self.raw_train, self.raw_valid, self.raw_test]
        for file in file_list:
            with open(file, "r") as fr:
                for line in fr.readlines():
                    tokens = re.split(r'\t|\s', line.strip())
                    head = tokens[0]
                    relation = tokens[1]
                    tail = tokens[2]

                    rel_dict[relation] = len(rel_dict)
                    ent_dict[head] = len(ent_dict)
                    ent_dict[tail] = len(ent_dict)
                    
        print("number of unique relations: %s" % len(rel_dict))
        print("number of unique entities: %s" % len(ent_dict))
        return ent_dict, rel_dict

    def preprocess(self):
        self.build_graph()
        self.build_query_data(self.raw_train, self.train)
        self.build_query_data(self.raw_test, self.test)
        self.build_query_data(self.raw_valid, self.valid)
        """ self.build_relation_context(self.raw_test, self.test_r_nbr)
        self.build_relation_context(self.raw_valid, self.valid_r_nbr) """
        self.build_relation_context(self.raw_train, self.ent_r_nbr)
        
        ent_list, rel_list = self.get_unique_roles_values()
        self.write_vocab(ent_list=ent_list, rel_list=rel_list)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        default=None,
        choices=[
            'fb237_v1', 'fb237_v2', 'fb237_v3', 'fb237_v4', 'nell_v1', 'nell_v2', 
            'nell_v3', 'nell_v4', 'WN18RR_v1', 'WN18RR_v2', 'WN18RR_v3', 'WN18RR_v4',
            'fb237_v1_ind', 'fb237_v2_ind', 'fb237_v3_ind', 'fb237_v4_ind', 'nell_v1_ind', 'nell_v2_ind', 
            'nell_v3_ind', 'nell_v4_ind', 'WN18RR_v1_ind', 'WN18RR_v2_ind', 'WN18RR_v3_ind', 'WN18RR_v4_ind'
        ])
    args = parser.parse_args()
    data_dir = os.path.join(os.getcwd(), 'data_preprocessed')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    task_dir = os.path.join(data_dir, args.task) #处理后的数据保存的路径
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    raw_train = os.path.join(os.getcwd(), 'data_raw', args.task, 'train.txt')
    raw_test = os.path.join(os.getcwd(), 'data_raw', args.task, 'test.txt')
    raw_valid = os.path.join(os.getcwd(), 'data_raw', args.task, 'valid.txt')

    train = os.path.join(task_dir, 'train.json')
    test = os.path.join(task_dir, 'test.json')
    valid = os.path.join(task_dir, 'valid.json')
    test_r_nbr = os.path.join(task_dir, 'test_r_nbr.json')
    valid_r_nbr = os.path.join(task_dir, 'valid_r_nbr.json')
    ent_r_nbr = os.path.join(task_dir, 'ent_r_nbr.json')

    neg_valid = os.path.join(task_dir, 'neg_valid.json')
    neg_test = os.path.join(task_dir, 'neg_test.json')

    rel_vocab = os.path.join(task_dir, 'vocab_rel.txt')
    ent_vocab = os.path.join(task_dir, 'vocab_ent.txt')

    data_process = DataProcessor(raw_train, raw_test, raw_valid, train, test, valid, 
            ent_vocab, rel_vocab, ent_r_nbr, neg_valid, neg_test)
    data_process.preprocess()
