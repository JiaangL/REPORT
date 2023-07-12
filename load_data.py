import json
from torch.utils.data import Dataset


def collate_fn(batch):
    batch = list(zip(*batch))
    [pos_id, relation, head, tail, path, num_path, path_mask, overall_mask] = batch
    del batch
    return pos_id, relation, head, tail, path, num_path, path_mask, overall_mask

def read_examples(data_path, vocab, relation_context, max_path_len, 
        real_max_num_path, is_sparse=False, filter_path=True, ablation=0):
    examples = []
    max_num_path = 0
    with open(relation_context, 'r') as f_dict:
        r_context = json.load(f_dict)
    with open(data_path, "r") as fr:
        for line in fr.readlines():
            obj = json.loads(line.strip())
            relation = obj['relation']
            head = obj['head']
            tail = obj['tail']
            paths = obj['path']
            num_path = obj['num_path']            
            positive_id = obj['positive_id']

            #while training, do not use samples without any paths
            if filter_path == True:
                if num_path == 0:
                    continue

            max_num_path = max(max_num_path, num_path)
            path_mask = []

            for it in paths:
                tmp_mask = [1 for i in range(len(it))]
                while len(it) < max_path_len:
                    it.append('[PAD]')
                    tmp_mask.append(0)
                 #add [CLS]
                it.insert(0, '[CLS]')
                tmp_mask.insert(0, 1)

                path_mask.append(tmp_mask)
                 
            #convert token to id
            relation = vocab.convert_tokens_to_ids([relation])[0]
            paths = [vocab.convert_tokens_to_ids(path) for path in paths]            
            head = vocab.convert_tokens_to_ids(r_context[head])
            tail = vocab.convert_tokens_to_ids(r_context[tail])
                  
            # create data examples
            example = [positive_id, relation, head, tail, paths, num_path, path_mask]    
            examples.append(example)

        for example in examples:
            if ablation == 0:
                # 4 for [MASK]\relation\head\tail
                overall_mask = [1] * (example[5] + 4) + [0] * (real_max_num_path - example[5])
            elif ablation == 1:
                #mask path
                overall_mask = [1] * 4 + [0] * real_max_num_path
            elif ablation == 2:
                #mask head & tail
                overall_mask = [1] * 2 + [0] * 2 + [1] * example[5] + [0] * (real_max_num_path - example[5])
            else:
                print('wrong ablation type!')
                raise
            # mask relation
            #overall_mask[1] = 0
            example.append(overall_mask)
            
    return examples, max_num_path

class MyDataset(Dataset):
    def __init__(self, data_path, vocab, relation_context, 
            max_path_len, real_max_num_path, is_sparse, filter_path=True, ablation=0):
        self.examples, self.max_num_path = read_examples(
            data_path, vocab, relation_context, max_path_len, real_max_num_path, filter_path=filter_path, ablation=ablation)
        self.length = len(self.examples)
    
    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return self.length
