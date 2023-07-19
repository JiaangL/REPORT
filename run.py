import logging
import argparse
import os
import torch
import numpy as np
import time
import re
import json
import math

from visualizer import get_local
#get_local.activate() # for visualizer

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import networkx as nx

from load_data import MyDataset, collate_fn
from reader.vocab_reader import Vocabulary
from model.gran_model import HierarchyTransformer
from preprocess_ind_data import filter_repeat_path
from sklearn.metrics import average_precision_score, roc_auc_score
from negative_sampling import convert_neg_sample
from reader.vocab_reader import Vocabulary

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(logger.getEffectiveLevel())

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=64)
parser.add_argument('--intermediate_size', type=int, default=128)
parser.add_argument('--path_transformer_layers', type=int, default=2)
parser.add_argument('--overall_transformer_layers', type=int, default=1)
parser.add_argument('--num_attention_heads', type=int, default=4)
parser.add_argument('--hidden_dropout_prob', type=float, default=0.4)
parser.add_argument('--path_attention_dropout_prob', type=float, default=0)
parser.add_argument('--overall_attention_dropout_prob', type=float, default=0)

parser.add_argument('--early_stop', type=int, default=15)
parser.add_argument('--eval_step', type=int, default=50)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument('--max_path_len', type=int, default=4)
parser.add_argument("--soft_label", type=float, default=1)
parser.add_argument("--margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")

parser.add_argument("--task", type=str, default='WN18RR_v1')
parser.add_argument('--sample_path', type=int, default=300)
parser.add_argument('--num_negative', type=int, default=1)
parser.add_argument('--max_num_path', type=int, default=10000) #v1 416  #v2 2967 #v4 6120
parser.add_argument('--cuda_id', type=int, default=3, help='GPU ID to use')
parser.add_argument('--checkpoints', type=str, default='ckpts')
parser.add_argument('--filter_empty_path', default='False', action='store_true', help='remove empty path instances while training?')
parser.add_argument('--max_rel_context', type=int, default=50)
parser.add_argument('--casual_path', default='False', action='store_true',
                        help='add additional casual path mask to path embeddings, and take output corresponding to the relation?')
parser.add_argument('--no_lr_scheduler', default='False', action='store_true',
                        help='Use LR scheduler?')
parser.add_argument('--encode_ent_pair', default='False', action='store_true', help='encode head and tail entity separately or together?')
parser.add_argument('--ablation', type=int, default=0, help='0: full model; 1: mask path; 2: mask relational context')

args = parser.parse_args()


def predict_grail_rank(model, device, neg_path, is_eval=False, ablation=0):
    data_dict = np.load(neg_path, allow_pickle=True).item()
    n_data = len(data_dict)
    #print('num of data: ', n_data)
    
    ranks = []
    ranks_empty = []

    for i in range(1, n_data + 1):
        neg_head_examples = data_dict[i]['neg_head_examples']
        neg_tail_examples = data_dict[i]['neg_tail_examples'] # type: list
        #print(len(neg_head_examples[0]))
        #print([it[0] for it in neg_head_examples][:4])

        #ablation study
        if is_eval == True and ablation > 0:
            n = len(neg_head_examples[-1])
            nn = len(neg_head_examples[-1][0])
            if ablation == 1: 
                for i in range(n):
                    neg_tail_examples[-1][i] = [1] * 4 + [0] * (nn - 4)
                    neg_head_examples[-1][i] = [1] * 4 + [0] * (nn - 4) #mask paths
            elif ablation == 2:
                for i in range(n):
                    neg_head_examples[-1][i][2:4] = [0] * 2 #mask head & tail
                    neg_tail_examples[-1][i][2:4] = [0] * 2
            else:
                print('wrong ablation type!')
                raise

        neg_head_scores = model(neg_head_examples, device)
        neg_tail_scores = model(neg_tail_examples, device)
        
        _, neg_head_results = torch.sort(neg_head_scores, descending=True)
        neg_head_tmp = neg_head_results == 0
        neg_head_rank = np.argwhere(neg_head_tmp.cpu().numpy()) + 1
        _, neg_tail_results = torch.sort(neg_tail_scores, descending=True)
        neg_tail_tmp = neg_tail_results == 0
        neg_tail_rank = np.argwhere(neg_tail_tmp.cpu().numpy()) + 1

        ranks.append(neg_head_rank)
        ranks.append(neg_tail_rank)

    ranks = np.asarray(ranks)
    ranks = ranks.ravel()
    hits10 = np.mean(ranks <= 10.0)
    hits5 = np.mean(ranks <= 5.0)
    hits1 = np.mean(ranks <= 1.0)
    mrr = np.mean(1.0 / ranks)
    print('hits10: %f , hits5: %f , hits1: %f , mrr: %f' % (hits10, hits5, hits1, mrr))

    return hits10



def auc_pr(model, device, neg_path, is_eval=False):
    data_dict = np.load(neg_path, allow_pickle=True).item()
    n = len(data_dict)
    
    rands = [5, 6]
    auc_pr_scores = []
    aucs = []
    test_batch_size = 128

    for rand in rands:
        pos_data, neg_data = [], []
        for pos_id in data_dict.keys():  
            if np.random.uniform() < 0.5:
                data = data_dict[pos_id]['neg_head_examples']
            else:
                data = data_dict[pos_id]['neg_tail_examples']

            if is_eval==True:
                n = len(data[-1])
                nn = len(data[-1][0])
                # for i in range(n):
                    #data[-1][i] = [1] * 4 + [0] * (nn - 4) #mask paths
                    # data[-1][i][2:4] = [0] * 2 #mask head & tail

            pos_data.append([data[i][0] for i in range(len(data))])  
            neg_data.append([data[i][rand] for i in range(len(data))])

        num_iteration = math.ceil(len(pos_data) / test_batch_size)

        for i in range(num_iteration):
            pos_data_tmp = pos_data[i * test_batch_size : (i + 1) * test_batch_size]
            neg_data_tmp = neg_data[i * test_batch_size : (i + 1) * test_batch_size]
            pos_data_convert_tmp = convert_neg_sample(pos_data_tmp, len(pos_data_tmp))
            neg_data_convert_tmp = convert_neg_sample(neg_data_tmp, len(pos_data_tmp))
            pos_data_convert_tmp[3] = list(pos_data_convert_tmp[3])
            neg_data_convert_tmp[3] = list(neg_data_convert_tmp[3])
        
            pos_fc_out_tmp = model(pos_data_convert_tmp, device)#[0:100]
            neg_fc_out_tmp = model(neg_data_convert_tmp, device)#[0:100]
            if i == 0:
                pos_fc_out = pos_fc_out_tmp
                neg_fc_out = neg_fc_out_tmp
            else:
                pos_fc_out = torch.cat((pos_fc_out, pos_fc_out_tmp), 0)
                neg_fc_out = torch.cat((neg_fc_out, neg_fc_out_tmp), 0)

        
        pos_labels = [1 for i in range(len(pos_fc_out))]
        neg_labels = [0 for i in range(len(pos_fc_out))]

        all_scores = torch.cat((pos_fc_out, neg_fc_out), 0).cpu()
        
        auc_pr_score = average_precision_score(pos_labels+neg_labels, all_scores)
        auc = roc_auc_score(pos_labels+neg_labels, all_scores)
        auc_pr_scores.append(auc_pr_score)
        aucs.append(auc)

    return aucs, auc_pr_scores


class TrainGraph(object):
    def __init__(self, raw_train, vocab, relation_context, neg_examples_path):
        self.raw_train = raw_train
        self.vocab = vocab
        self.neg_examples = np.load(neg_examples_path, allow_pickle=True).item()
        with open(relation_context, 'r') as f_dict:
            self.r_context = json.load(f_dict)

    def build_graph(self):
        self.biG = nx.MultiDiGraph()
        total_links = 0
        with open(self.raw_train, 'r') as f_train:
            for line in f_train.readlines():
                total_links = total_links + 1
                tokens = re.split(r'\t|\s', line.strip())
                head = tokens[0]
                tail = tokens[2]
                relation = tokens[1]

                self.biG.add_edge(head, tail, **{'r_type': relation})
                self.biG.add_edge(tail, head, **{'r_type': 'inv_' + relation})
        f_train.close()

    def neg_selection_for_training(self, train_batch, num_all_sample=200, num_negative=1):
        #neg_head_batch = []
        #neg_tail_batch = []
        batch_size = len(train_batch[0])
        if num_negative == 1:
            neg_batch = []
            for i in range(batch_size):
                example = [train_batch[j][i] for j in range(len(train_batch))]
                
                head = example[2]
                tail = example[3]
                relation = example[1]
                positive_id = example[0]

                n_neg_examples = len(self.neg_examples[positive_id])
                neg_example = [self.neg_examples[positive_id][j][1] for j in range(n_neg_examples)]
                
                neg_batch.append(neg_example)
            
            neg_batch = convert_neg_sample(neg_batch, len(neg_batch))
        else:
            neg_batch = [ [] for x in range(num_negative) ]
            for num in range(num_negative):
                for i in range(batch_size):
                    example = [train_batch[j][i] for j in range(len(train_batch))]
                    positive_id = example[0]
                    
                    n_neg_examples = len(self.neg_examples[positive_id])
                    neg_example = [self.neg_examples[positive_id][j][num + 1] for j in range(n_neg_examples)]
                    
                    neg_batch[num].append(neg_example)
                
                neg_batch[num] = convert_neg_sample(neg_batch[num], len(neg_batch[num]))

        return neg_batch

    def neg_sampling_for_training(self, train_batch):
        # create two negative samples for each positive sample, each replacing head or tail
        train_batch = train_batch[1:]
        batch_size = len(train_batch[0])
        
        node_list = list(self.biG.nodes) # the range for sampling negative samples
        n = len(node_list)
        examples = []
        neg_head_examples = []
        neg_tail_examples = []
        for i in range(batch_size):
            example = [train_batch[j][i] for j in range(len(train_batch))]
            head = example[1]
            tail = example[2]
            relation = example[0]
            path_id = [self.vocab.convert_tokens_to_ids(path) for path in example[3]]
            relation_id = self.vocab.convert_tokens_to_ids([relation])[0]
            
            #creat negative head examples
            while 1:
                neg_head = node_list[np.random.choice(n)]
                flag = 1
                if neg_head != tail:
                    #filter positive candidates
                    if self.biG.has_edge(neg_head, tail):
                        for i in self.biG[neg_head][tail]:
                            if self.biG[neg_head][tail][i]['r_type'] == relation:
                                flag = 0
                                break #neg_head is actually a positive sample
                    if flag == 1:
                        flag = 2# find a real negative sample
                        break
            
            neg_head_paths = nx.all_simple_edge_paths(self.biG, neg_head, tail, 4)
            neg_head_relation_paths = []
            for path in neg_head_paths:
                relation_path = [self.biG[edge[0]][edge[1]][edge[2]]
                    ['r_type'] for edge in path]
                neg_head_relation_paths.append(relation_path)

            neg_head_relation_paths = filter_repeat_path(neg_head_relation_paths)
            tmp = []
            neg_head_pathmask = []
            for path in neg_head_relation_paths:
                tmp_mask = [1 for i in range(len(path))]
                while len(path) < args.max_path_len:
                    path.append('[PAD]')
                    tmp_mask.append(0)
                path.insert(0, '[CLS]')
                tmp_mask.insert(0, 1)
                neg_head_pathmask.append(tmp_mask)
                tmp.append(self.vocab.convert_tokens_to_ids(path))
            neg_head_relation_paths = tmp
            
            neg_head_example = [relation_id, 
                    self.vocab.convert_tokens_to_ids(self.r_context[neg_head]), 
                    self.vocab.convert_tokens_to_ids(self.r_context[tail]), 
                    neg_head_relation_paths,
                    len(neg_head_relation_paths), 
                    neg_head_pathmask]
            neg_head_overall_mask = [1] * (neg_head_example[4] + 4) \
                + [0] * (args.max_num_path - neg_head_example[4]) # 4 for mask\r\h\t
            neg_head_example.append(neg_head_overall_mask)

            #creat negative tail examples
            while 1:
                neg_tail = node_list[np.random.choice(n)]
                flag = 1
                if neg_tail != head:
                    #filter positive candidates
                    if self.biG.has_edge(head, neg_tail):
                        for i in self.biG[head][neg_tail]:
                            if self.biG[head][neg_tail][i]['r_type'] == relation:
                                flag = 0
                                break #neg_head is actually a positive sample
                    if flag == 1:
                        flag = 2# find a real negative sample
                        break

            neg_tail_paths = nx.all_simple_edge_paths(self.biG, head, neg_tail, 4)
            neg_tail_relation_paths = []
            for path in neg_tail_paths:
                relation_path = [self.biG[edge[0]][edge[1]][edge[2]]
                    ['r_type'] for edge in path]
                neg_tail_relation_paths.append(relation_path)

            neg_tail_relation_paths = filter_repeat_path(neg_tail_relation_paths)
            tmp = []
            neg_tail_pathmask = []
            for path in neg_tail_relation_paths:
                tmp_mask = [1 for i in range(len(path))]
                while len(path) < args.max_path_len:
                    path.append('[PAD]')
                    tmp_mask.append(0)
                path.insert(0, '[CLS]')
                tmp_mask.insert(0, 1)
                neg_tail_pathmask.append(tmp_mask)
                tmp.append(self.vocab.convert_tokens_to_ids(path))
            neg_tail_relation_paths = tmp
            
            neg_tail_example = [relation_id, 
                    self.vocab.convert_tokens_to_ids(self.r_context[head]), 
                    self.vocab.convert_tokens_to_ids(self.r_context[neg_tail]), 
                    neg_tail_relation_paths,
                    len(neg_tail_relation_paths), 
                    neg_tail_pathmask]
            neg_tail_overall_mask = [1] * (neg_tail_example[4] + 4) \
                + [0] * (args.max_num_path - neg_tail_example[4])
            neg_tail_example.append(neg_tail_overall_mask)

            example_convert = [relation_id,
                    self.vocab.convert_tokens_to_ids(self.r_context[head]), 
                    self.vocab.convert_tokens_to_ids(self.r_context[tail]),
                    path_id, len(path_id), example[5], example[6]]
            examples.append(example_convert)
            neg_head_examples.append(neg_head_example)
            neg_tail_examples.append(neg_tail_example)

        neg_head_examples = convert_neg_sample(neg_head_examples, len(examples))
        neg_tail_examples = convert_neg_sample(neg_tail_examples, len(examples))
        examples = convert_neg_sample(examples, len(examples))
        
       
        return examples, neg_head_examples, neg_tail_examples


def main(args):
    config = vars(args)
    ablation = config['ablation']
    device = torch.device("cuda", args.cuda_id)
    args.intermediate_size = 2 * args. embedding_size
    args.filter_empty_path = True
    print('no_lr_scheduler? ', args.no_lr_scheduler)
    print('casual path? ', args.casual_path)
    print('learning rate', args.learning_rate)
    print('filter empty path? ', args.filter_empty_path)
    print('number of negative: ', args.num_negative)
    print('encode entity pair? ', args.encode_ent_pair)

    #relation名称和id的对应关系
    vocabulary_relation = Vocabulary(vocab_file=args.vocab_path)
    is_sparse = True if args.task=='nell_v1' else False

    if args.do_train:
        train_data = MyDataset(args.train_file, vocabulary_relation, args.r_context_all,
                args.max_path_len, args.sample_path, is_sparse=is_sparse, filter_path=args.filter_empty_path, ablation=ablation)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, 
                            shuffle=True, collate_fn=collate_fn)

        train_KG = TrainGraph(args.raw_train, vocabulary_relation, 
                args.r_context_all, neg_examples_path=args.train_neg_examples)
        train_KG.build_graph()
        args.vocab_relation_size = len(vocabulary_relation.vocab)
        
        num_train_instances = train_data.length
        warmup_epochs = int(args.epoch * args.warmup_proportion)

        logger.info("Num train instances: %d" % num_train_instances)
        logger.info("Train epochs: %d" % args.epoch)
        logger.info("Train warmup steps: %d" % warmup_epochs)

        model = HierarchyTransformer(config=config, device=device)

    if args.do_predict:
        predict_data = MyDataset(args.train_file, vocabulary_relation, 
                args.r_context_all, args.max_path_len, args.max_num_path)

    if args.do_test:
        test_data = MyDataset(args.test_file, vocabulary_relation, args.r_context_test, 
                args.max_path_len, args.sample_path, is_sparse=is_sparse, filter_path=args.filter_empty_path)

    if args.do_train:
        model.train()
        model = model.to(device)
        model_params = list(model.parameters())
        print('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-6, weight_decay=args.weight_decay)
        lambda1 = lambda epoch: min((epoch+1)/warmup_epochs,(args.epoch-epoch-1)/(args.epoch-warmup_epochs))
        optim_schedule = LambdaLR(opt, lr_lambda=lambda1, last_epoch=-1)

        best_metric = 0
        best_test_hits10 = 0
        best_test_ap = 0
        n_not_growing_epoch = 0
        step = 0

        for epoch in range(1, args.epoch+1):
            time_begin = time.time()
            train_loss_per_epoch = 0
            n_not_growing_epoch = n_not_growing_epoch + 1
            for batch in train_loader:
                step = step + 1
                neg_batch= train_KG.neg_selection_for_training(batch, num_negative=args.num_negative)     
                
                pos_score = model(batch, device)
                neg_score = model(neg_batch, device)
                
                loss = model.get_bce_loss(pos_score, neg_score, device, num_negative=args.num_negative)
                train_loss_per_epoch += loss
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            print("第%d个epoch的学习率：%f" % (epoch, opt.param_groups[0]['lr']))
            optim_schedule.step()
            time_end = time.time()
            
            train_loss_per_epoch = train_loss_per_epoch / num_train_instances
            used_time = time_end - time_begin
            logger.info("epoch: %d, train_loss:%f, time cost:%f (s)" %
                        (epoch, train_loss_per_epoch, used_time))
            if epoch == args.epoch or epoch % 20 == 0: #or epoch<=20 and epoch%2==0:
                save_path = os.path.join(args.checkpoints, args.task, "epoch_" + str(epoch))
                torch.save(model, save_path)
                logger.info("Model saved at epoch %d" % (epoch))

            model.eval()
            with torch.no_grad():     
                aucs, auc_pr_scores = auc_pr(model=model, neg_path=args.neg_save_path_valid, device=device, is_eval=True) #args.neg_save_path_valid
                auc_std = np.std(aucs)
                auc_pr_std = np.std(auc_pr_scores)
                auc = np.mean(aucs)
                auc_pr_score = np.mean(auc_pr_scores)
               
                if auc > best_metric:
                #if True:
                    print('auc \ auc_pr on valid; best auc so far:', auc, auc_pr_score, best_metric)
                    print('auc_std \ auc_pr_std is:', auc_std, auc_pr_std)
                    n_not_growing_epoch = 0
                    best_metric = auc
                    save_path = os.path.join(args.checkpoints, args.task, 
                        str(args.embedding_size)+'_'+str(args.learning_rate)+'_'+str(args.batch_size)
                        +'_'+str(args.num_attention_heads)+'_'+"best_yet")
                    # create the path if not exists
                    if not os.path.exists(save_path):
                        os.makedirs(os.path.join(args.checkpoints, args.task))
                    torch.save(model, save_path)
                    print('Better models found w.r.t auc_roc. Saved it!')

                    aucs, auc_pr_scores = auc_pr(model=model, neg_path=args.neg_save_path_test, device=device)
                    auc_std = np.std(aucs)
                    auc_pr_std = np.std(auc_pr_scores)
                    auc = np.mean(aucs)
                    auc_pr_score = np.mean(auc_pr_scores)
                    print(f'results for auc_pr on test is: {auc_pr_score}, auc_pr_std is: {auc_std}')

                    print('results for grail rank metrics on test:')
                    eval_performance = predict_grail_rank(
                            model=model, device=device, neg_path=args.neg_save_path_test, is_eval=True, ablation=ablation)
                    best_test_ap = auc_pr_score
                    best_test_hits10 = 0
                    best_test_hits10 = eval_performance
            model.train()

            #Early Stopping
            if n_not_growing_epoch == args.early_stop:
                break

        print(f'Best test auc-pr is: {best_test_ap}! Best test hits@10 is: {best_test_hits10}!')            

    if args.do_predict:
        print('Predict on Valid Set')
        #select_epochs = [x * 2 for x in range(1, 10)] + [20, 40, 60, 80, 100]
        select_epochs = [10]
        for epoch in select_epochs:
            
            if not args.do_train:
                save_path = os.path.join(args.checkpoints, args.task, "epoch_" + str(epoch))
                model = torch.load(save_path, map_location='cuda:4').to(device)
            model.eval()
            print('-----------eval epoch: ', epoch, '-------------')
            with torch.no_grad():
                
                print('results for grail rank metrics')
                eval_performance = predict_grail_rank(
                        model=model, device=device, neg_path=args.neg_save_path_valid)

    if args.do_test:
        print('Predict on Test Set')
        save_path = os.path.join(args.checkpoints, args.task, 
                str(args.embedding_size)+'_'+str(args.learning_rate)+'_'+
                str(args.batch_size) +'_'+str(args.num_attention_heads)+'_'+"best_yet")
        # save_path = os.path.join(args.checkpoints, args.task, '2560.0011284best_yet')
        model = torch.load(save_path, map_location=device).to(device)

        model.eval()
        model_params = list(model.parameters())
        print('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))
        print('results for grail rank metrics')
        t1 = time.time()
        eval_performance = predict_grail_rank(
                model=model, device=device, neg_path=args.neg_save_path_test)
        t2=time.time()
        print('Running tims is: %s seconds' % ((t2 - t1)*1))

        """ Explaination of REPORT """
        # data_dict = np.load(args.neg_save_path_test, allow_pickle=True).item()
        # pick_data = [2]
        # positive_data = []
        # for i in pick_data:
        #     neg_head_examples = data_dict[i]['neg_head_examples']
        #     tmp = [[it[0]] for it in neg_head_examples]
        #     positive_data.append(tmp)
        #     print(tmp[:-2])
        #     score = model(tmp, device)

        # cache = get_local.cache
        # attention_maps = cache['multi_head_attention.compute_attention']

        if args.task[-3:] == 'ind':
            vocab_path = os.path.join('data_preprocessed', args.task[:-4], 'vocab_rel.txt')
        else:
            vocab_path = os.path.join('data_preprocessed', args.task, 'vocab_rel.txt')
        vocab = Vocabulary(vocab_file=vocab_path)

        """ Explaination of REPORT (cont.) """
        # for head_id in range(4):
        #     relation_weights = attention_maps[4][:,head_id,1,:]
        #     print('attention head:', head_id)
        #     for i in range(len(pick_data)):
        #         weight = relation_weights[i]
        #         path = positive_data[i][3][i]
        #         sort_weights_index = np.argsort(weight)[::-1]
        #         sort_weights = np.sort(weight)[::-1]
        #         select_weights_index = list(sort_weights_index[:5])
        #         select_weights = sort_weights[:5]

        #         select_item = [path[it] for it in select_weights_index]
        #         print(select_weights)
        #         print(select_weights_index)
        #         print(select_item)
                

if __name__ == '__main__':
    #print_arguments(args)
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args.do_train = True
    args.do_predict = False
    args.do_test = True
    args.raw_train = os.path.join(os.getcwd(), 'data_raw', args.task, 'train.txt')
    args.raw_test = os.path.join(os.getcwd(), 'data_raw', args.task, 'test.txt')
    args.raw_valid = os.path.join(os.getcwd(), 'data_raw', args.task, 'valid.txt')

    args.vocab_path = os.path.join('data_preprocessed', args.task, 'vocab_rel.txt')
    args.inv_train_file = os.path.join('data_preprocessed', args.task+'_ind', 'train.json')
    args.train_file = os.path.join('data_preprocessed', args.task, 'train.json')
    args.valid_file = os.path.join('data_preprocessed', args.task, 'valid.json')
    args.test_file = os.path.join('data_preprocessed', args.task+'_ind', 'test.json')
    args.train_neg_examples = os.path.join('neg_samples', args.task, 'neg_sample_train.npy')

    #train KG里所有节点的relation context
    args.r_context_all = os.path.join('data_preprocessed', args.task, 'ent_r_nbr.json')
    args.r_context_train = os.path.join('data_preprocessed', args.task, 'test_r_nbr.json')
    args.r_context_valid = os.path.join('data_preprocessed', args.task, 'valid_r_nbr.json')
    args.r_context_test = os.path.join('data_preprocessed', args.task+'_ind', 'ent_r_nbr.json')

    args.neg_save_path_train = os.path.join('neg_samples', args.task, 'neg_sample_train.npy')
    args.neg_save_path_valid = os.path.join('neg_samples', args.task, 'neg_sample_valid.npy')
    args.neg_save_path_test = os.path.join('neg_samples', args.task+'_ind', 'neg_sample_test.npy')
    
    
    main(args)
