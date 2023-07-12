import torch.nn as nn
import torch


def truncated_normal_(tensor,mean=0,std=0.02):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


def weight_init(m):
    if isinstance(m, nn.Linear):
        truncated_normal_(m.weight.data)
        m.bias.data.zero_()
    """ elif isinstance(m, nn.Embedding):
        truncated_normal_(m.weight.data) """