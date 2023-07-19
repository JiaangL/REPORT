import logging
import torch
import math
import torch.nn as nn
import torch.nn.parameter as parameter
import torch.nn.functional as F
import numpy as np
import copy
import random

from torch.autograd import Variable
from model.graph_encoder import encoder

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())


def SoftCrossEntropy(inputs, target, reduction='average'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    elif reduction == 'sum':
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


class HierarchyTransformer(nn.Module):
    """
    Hierarchical Tranformer model class
    """
    def __init__(self,
                 config,
                 device):
        super(HierarchyTransformer, self).__init__()
        self._n_path_layers = config['path_transformer_layers']
        self._n_overall_layers = config['overall_transformer_layers']
        self._n_head = config['num_attention_heads']
        self._emb_size = config['embedding_size']
        self._intermediate_size = config['intermediate_size']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._path_attention_dropout = config['path_attention_dropout_prob']
        self._overall_attention_dropout = config['overall_attention_dropout_prob']

        self._voc_size = config['vocab_relation_size']
        self._max_path_len = config['max_path_len']
        self.sample_path = config['sample_path']
        self._max_num_path = config['max_num_path'] if self.sample_path < 0 else self.sample_path

        self._soft_label = config['soft_label']
        self._batch_size = config['batch_size']

        self.device = device
        self.margin = config['margin']
        self.max_rel_context = config['max_rel_context']
        self.casual_path = config['casual_path']
        self.is_ent_pair = config['encode_ent_pair']

        self.path_transformer = encoder(
            n_layer=self._n_path_layers,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._intermediate_size,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._path_attention_dropout,
            relu_dropout=0,
            device=device,
            max_seq_len=self._max_path_len+1)#2 for [CLS] and relaion

        self.overall_transformer = encoder(
            n_layer=self._n_overall_layers,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._intermediate_size,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._overall_attention_dropout,
            relu_dropout=0,
            device=device,
            max_seq_len=self._max_num_path+4) #relation\head\tail\[CLS]

        self.entity_transformer = encoder(
            n_layer=self._n_path_layers,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._intermediate_size,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._overall_attention_dropout,
            relu_dropout=0,
            device=device,
            max_seq_len=self.max_rel_context+1
        )

        self.emb_look_up = nn.Embedding(self._voc_size, self._emb_size)
        self.emb_look_up_in_path = nn.Embedding(self._voc_size, self._emb_size)
        self.emb_look_up_in_nbr = nn.Embedding(self._voc_size, self._emb_size)
        self.positoin_emb_look_up = nn.Embedding(self._max_path_len+2, self._emb_size) #2 for [CLS] and relation
        self.type_emb_look_up = nn.Embedding(4, self._emb_size) #relation\head\tail\path

        self.mask_trans_linear = nn.Linear(self._emb_size, self._emb_size, bias=True)
        self.output_score_linear = nn.Linear(self._emb_size, 1, bias=True)
        self.head_trans_linear = nn.Linear(self._emb_size, self._emb_size, bias=True)
        self.tail_trans_linear = nn.Linear(self._emb_size, self._emb_size, bias=True)
        self.ent_trans_linear = nn.Linear(self._emb_size, self._emb_size, bias=True)
        self.bn1 = nn.BatchNorm1d(self._max_path_len+1, affine=True)
        self.bn2 = nn.BatchNorm1d(self._emb_size, affine=True)
        self.bn3 = nn.BatchNorm1d(self._max_num_path+4, affine=True) #4 for relation\head\tail\[MASK]
        self.bn4 = nn.BatchNorm1d(self._emb_size, affine=True) #for head&tail transform
        self.bn5 = nn.BatchNorm1d(self._emb_size, affine=True)
        self.bn6 = nn.BatchNorm1d(self.max_rel_context+1, affine=True)
        self.ln1 = nn.LayerNorm(self._emb_size)
        self.ln2 = nn.LayerNorm(self._emb_size)
        self.ln3 = nn.LayerNorm(self._emb_size)
        self.ln4 = nn.LayerNorm(self._emb_size)
        self.ln5 = nn.LayerNorm(self._emb_size)
        self.ln6 = nn.LayerNorm(self._emb_size)
        self.ln7 = nn.LayerNorm(self._emb_size)
        self.dropout = nn.Dropout(self._prepostprocess_dropout)
        self.fc_out_bias = nn.Parameter(torch.empty(self._voc_size), requires_grad=True)
        self.device = device

        nn.init.zeros_(self.fc_out_bias)

    def get_socre(self):
        self.overall_enc_out = self.ln5(self.overall_enc_out)
        self.overall_enc_out = self.dropout(self.overall_enc_out)
        mask_trans_feat = self.mask_trans_linear(self.overall_enc_out)

        triplet_score = self.output_score_linear(F.gelu\
            (self.ln2(mask_trans_feat))).squeeze(1)
        
        return triplet_score

    
    def get_bce_loss(self, pos_score, neg_score, device, num_negative=1):
        bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        pos_labels = torch.ones(pos_score.shape[0]).to(device)
        neg_labels = torch.zeros(pos_score.shape[0]).to(device)
        all_labels = torch.cat((pos_labels, neg_labels), 0)
        
        all_labels = torch.cat((pos_labels, neg_labels), 0)
        all_bce_loss = bce_loss(torch.cat((pos_score, neg_score), 0), all_labels)

        return all_bce_loss


    def path_representation(self, path, path_mask, device):
        tmp = torch.tensor(path, dtype=torch.int).to(device)
        path_emb = self.emb_look_up(tmp)   

        # add position embedding to paths
        all_pos = [x for x in range(self._max_path_len + 1)] #2 for [CLS] and relation
        all_pos = torch.tensor(all_pos).to(device)
        path_position_emb = self.positoin_emb_look_up(all_pos)
        path_merge_emb += path_position_emb
        #mask
        path_mask = torch.tensor(path_mask, dtype=torch.float).to(device)
        path_mask = torch.unsqueeze(path_mask, -1)
        path_attn_mask = torch.matmul(path_mask, path_mask.transpose(-1, -2))           
        path_attn_mask = torch.mul(torch.sub(path_attn_mask, 1.0), 1000000.0)   
        n_head_path_attn_mask = torch.stack([path_attn_mask] * self._n_head, axis=1)         
        n_head_path_attn_mask.requires_gradient = False
        
        # preprocess
        path_emb = self.ln1(path_emb)
        path_emb = self.dropout(path_emb)
        
        #Path Transformer to obtain path representations
        if not (len(path_emb.shape) == 3):
            print(path_emb.shape)
            raise ValueError(
                "Inputs: path_emb should be 3-D tensors.")
        _path_enc_out = self.path_transformer(
            enc_input=path_emb,
            attn_bias=n_head_path_attn_mask)
        test_path_enc_out = torch.index_select(_path_enc_out, dim=1, 
            index=torch.tensor(0).to(device)).squeeze(1) # take [CLS] as output

    def visualize(self, sample, device):
        pass

    def forward(self, batch, device):
        #[relation, head, tail, path, num_path, path_mask, overall_mask] = batch
        if len(batch) == 7:
            [relation, head, tail, path, num_path, path_mask, overall_mask] = batch
            pos_id = None
        elif len(batch) == 8:
            [pos_id, relation, head, tail, path, num_path, path_mask, overall_mask] = batch
        else:
            print(batch)
            raise

        #randomly sample relational paths between entities
        if self.sample_path >= 0:
            path = list(path)
            path_mask = list(path_mask)
            overall_mask = list(overall_mask)

            for i in range(len(path)):
                n = num_path[i]
                if n > self.sample_path:
                    if pos_id is not None:
                        #random.seed(pos_id[i])
                        rand_id = random.sample([x for x in range(n)], self.sample_path)
                    else:
                        rand_id = random.sample([x for x in range(n)], self.sample_path)
                    path[i] = [path[i][id_] for id_ in rand_id]
                    path_mask[i] = [path_mask[i][id_] for id_ in rand_id]
                overall_mask[i] = overall_mask[i][ : 4 + self.sample_path]
                overall_mask[i][0] = 0 # fusion module does not consider [MASK]

            num_path = [min(it, self.sample_path) for it in num_path]
        
        path_merge = list()
        path_mask_merge = list()
        all_pos = [x for x in range(self._max_path_len + 1)] #2 for [CLS] and relation
        all_pos = torch.tensor(all_pos).to(device)
        path_position_emb = self.positoin_emb_look_up(all_pos)

        for its in path:       
            path_merge.extend(its)
        for its in path_mask:
            path_mask_merge.extend(its)

        accu_num_path = torch.tensor([0] + list(num_path), dtype=torch.int).to(device)
        accu_num_path = torch.cumsum(accu_num_path, dim=0)
        num_path = torch.tensor(num_path, dtype=torch.float32).to(device)

        mask_emb = self.emb_look_up(torch.tensor([2]*len(head)).to(device)) #[MASK]

        #Entity Transformer
        max_n_rel_nbr = max([len(it) for it in head] + [len(it) for it in tail])
        
        if self.is_ent_pair is True:
            pad_head = [[3] + it + [0] * (max_n_rel_nbr - len(it)) for it in head] #inv_[MASK] id
            pad_tail = [it + [0] * (max_n_rel_nbr - len(it)) for it in tail] #inv_[CLS] id
            pad_ent_pair = [it_h + it_t for (it_h, it_t) in zip(pad_head, pad_tail)]
            ent_attn_bias = [[1] * len(it_h) + [0] * (max_n_rel_nbr - len(it_h)) + [1] * len(it_t) + [0] * (max_n_rel_nbr - len(it_t))\
                 for (it_h, it_t) in zip(pad_head, pad_tail)]
        else:
            pad_head = [[3] + it + [0] * (max_n_rel_nbr - len(it)) for it in head] 
            pad_tail = [[5] + it + [0] * (max_n_rel_nbr - len(it)) for it in tail] 
            ent_attn_bias = [[1] * (len(it) + 1) + [0] * (max_n_rel_nbr - len(it)) for it in head]\
                + [[1] * (len(it) + 1) + [0] * (max_n_rel_nbr - len(it)) for it in tail]
        
        ent_attn_bias = torch.tensor(ent_attn_bias, dtype=torch.float).to(device).unsqueeze(-1)
        ent_attn_mask = torch.matmul(ent_attn_bias, ent_attn_bias.transpose(-1, -2))
        ent_attn_mask = torch.mul(torch.sub(ent_attn_mask, 1.0), 10000.0)
        n_head_ent_attn_mask = torch.stack([ent_attn_mask] * self._n_head, axis=1)
        n_head_ent_attn_mask.requires_gradient = False

        if self.is_ent_pair is True:
            head_tail_emb_input = self.emb_look_up(torch.tensor(pad_ent_pair, \
                dtype=torch.int).to(device))
            type_id = [0] + [1 for x in range(max_n_rel_nbr)] + [2 for x in range(max_n_rel_nbr)]
            head_tail_emb_input = head_tail_emb_input + self.type_emb_look_up(torch.tensor(type_id, dtype=torch.int).to(device))
        else:
            head_tail_emb_input = self.emb_look_up(torch.tensor(pad_head + pad_tail, \
                dtype=torch.int).to(device))

        head_tail_emb_input = self.ln6(head_tail_emb_input)
        head_tail_emb_input = self.dropout(head_tail_emb_input)
        _head_tail_enc_out  = self.entity_transformer(
            enc_input=head_tail_emb_input, attn_bias=n_head_ent_attn_mask
        )
        
        self.head_tail_enc_out = torch.index_select(_head_tail_enc_out, dim=1, 
                index=torch.tensor(0).to(device)).squeeze(1) #take [CLS] from entity transformer as output
        
        if self.is_ent_pair is True:
            ent_pair_emb = self.head_tail_enc_out
        else:
            head_emb = self.head_tail_enc_out[:len(head)]
            tail_emb = self.head_tail_enc_out[len(head):]

        r_emb = self.emb_look_up(torch.tensor(relation, dtype=torch.int).to(device))

        if len(path_merge) == 0:
            #A batch with no valid relational path, which occurs in evaluation
            self.path_enc_out = torch.empty(len(head), self._emb_size)
        else:
            tmp = torch.tensor(path_merge, dtype=torch.int).to(device)
            path_merge_emb = self.emb_look_up(tmp)   

            # add position embedding to paths
            path_merge_emb += path_position_emb
            # due with mask
            path_mask_merge = torch.tensor(path_mask_merge, dtype=torch.float).to(device)
            path_mask_merge = torch.unsqueeze(path_mask_merge, -1)
            path_attn_mask = torch.matmul(path_mask_merge, path_mask_merge.transpose(-1, -2))#???                
            path_attn_mask = torch.mul(torch.sub(path_attn_mask, 1.0), 1000000.0) 

            n_head_path_attn_mask = torch.stack([path_attn_mask] * self._n_head, axis=1)         
            n_head_path_attn_mask.requires_gradient = False
            
            # preprocess
            path_merge_emb = self.ln1(path_merge_emb)
            path_merge_emb = self.dropout(path_merge_emb)
            
            #Path Transformer to obtain representations of relational paths
            if not (len(path_merge_emb.shape) == 3):
                print(path_merge_emb.shape)
                raise ValueError(
                    "Inputs: path_merge_emb should be 3-D tensors.")
            _path_enc_out = self.path_transformer(
                enc_input=path_merge_emb,
                attn_bias=n_head_path_attn_mask)
            self.path_enc_out = torch.index_select(_path_enc_out, dim=1, 
                    index=torch.tensor(0).to(device)).squeeze(1) #[CLS] as output
                
        #overall transformer, which fuses relational paths and context
        if self.is_ent_pair is True:
            for i in range(len(overall_mask)):
                overall_mask[i] = overall_mask[i][1:]

        overall_mask_emb = self.emb_look_up(torch.tensor(overall_mask).to(device))
        overall_mask = torch.tensor(overall_mask, dtype=torch.float).to(device)
        overall_mask = torch.unsqueeze(overall_mask, -1)
        overall_attn_mask = torch.matmul(overall_mask, overall_mask.transpose(-1, -2))
        overall_attn_mask = torch.mul(torch.sub(overall_attn_mask, 1.0), 1000000.0)
        n_head_overall_attn_mask = torch.stack([overall_attn_mask] * self._n_head, axis=1)
        n_head_overall_attn_mask.requires_gradient = False

        max_path_input = torch.empty(len(head), self._max_num_path, self._emb_size).to(device)
        for i in range(len(head)):
            if num_path[i] == 0:
                if self.is_ent_pair is True:
                    max_path_input[i] = overall_mask_emb[i, 3:]
                else:
                    max_path_input[i] = overall_mask_emb[i, 4:]
            else:
                path_pad_input = self.path_enc_out[accu_num_path[i]:accu_num_path[i+1]]
                if self.is_ent_pair is True:
                    max_path_input[i] = torch.cat((path_pad_input, overall_mask_emb[i, num_path[i].int()+3:]), 0)
                else:
                    max_path_input[i] = torch.cat((path_pad_input, overall_mask_emb[i, num_path[i].int()+4:]), 0)
        
        if self.is_ent_pair is True:
            overall_transformer_input = torch.cat((mask_emb.unsqueeze(1), r_emb.unsqueeze(1),
                ent_pair_emb.unsqueeze(1), max_path_input), 1)
        else:
            overall_transformer_input = torch.cat((mask_emb.unsqueeze(1), r_emb.unsqueeze(1),
                head_emb.unsqueeze(1), tail_emb.unsqueeze(1), max_path_input), 1)

        overall_transformer_input = self.ln3(overall_transformer_input)
        overall_transformer_input = self.dropout(overall_transformer_input)
        _overall_enc_out = self.overall_transformer(
            enc_input=overall_transformer_input,
            attn_bias=n_head_overall_attn_mask)
        self.overall_enc_out = torch.index_select(_overall_enc_out, dim=1, 
            index=torch.tensor(1).to(device)).squeeze(1) #1 for relation
        
        triplet_score = self.get_socre()

        return triplet_score

