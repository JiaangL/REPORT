"""Graph Transformer encoder."""

import torch
import torch.nn as nn
import torch.nn.parameter as parameter
import torch.nn.functional as F
from visualizer import get_local


class multi_head_attention(torch.nn.Module):
    def __init__(self,
                d_key,
                d_value,
                d_model,
                n_head=1,
                dropout_rate=0.):
        super(multi_head_attention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_key * n_head, bias=True)
        self.W_K = nn.Linear(d_model, d_key * n_head, bias=True)
        self.W_V = nn.Linear(d_model, d_value * n_head, bias=True)
        self.output_linear = nn.Linear(d_key * n_head, d_model)
        
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.n_head = n_head 
        self.dropout_rate = dropout_rate

        self.weight_dropout = nn.Dropout(dropout_rate)


    def __compute_qkv(self, queries, keys, values):
        """
        Add linear projection to queries, keys, and values.
        """
        q = self.weight_dropout(self.W_Q(queries))
        k = self.weight_dropout(self.W_K(keys))
        v = self.weight_dropout(self.W_V(values))

        return q, k, v

    def __split_heads(self, x, n_head):
        """
        Reshape the last dimension of input tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = torch.reshape(
            x, [x.shape[0], x.shape[1], n_head, hidden_size // n_head])

        trans = reshaped.transpose(1, 2)
        # permute the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return trans

    def __combine_heads(self, x):
        """
        Transpose and then reshape the last two dimensions of input tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        x = x.transpose(1, 2)
        reshaped = torch.reshape(x, [x.shape[0], x.shape[1], x.shape[2]*x.shape[3]])
        return reshaped

    #@get_local('weights')
    def compute_attention(self, q, k, v, attn_bias):
        """
        Edge-aware Self-Attention.

        Scalar dimensions referenced here:
            B = batch_size
            M = max_sequence_length
            N = num_attention_heads
            H = hidden_size_per_head

        Args:
            q: reshaped queries [B, N, M, H]
            k: reshaped keys    [B, N, M, H]
            v: reshaped values  [B, N, M, H]
            edges_k: edge representations between input tokens (keys)   [M, M, H]
            edges_v: edge representations between input tokens (values) [M, M, H]
            attn_bias: attention mask [B, N, M, M]
        """
        if not (len(q.shape) == len(k.shape) == len(v.shape) == 4):
            raise ValueError("Input q, k, v should be 4-D Tensors.")
         # regular self-attention
        scaled_q = torch.mul(q, self.d_key**-0.5)
        product = torch.matmul(scaled_q, k.transpose(-1, -2))
        # add attention bias        
        if attn_bias is not None:
            product += attn_bias
        # softmax attention weights
        weights = nn.Softmax(dim=-1)(product)
        """ if self.dropout_rate:
            weights = self.weight_dropout(weights) """
        out = torch.matmul(weights, v)

        return out

    def forward(self, queries, keys, values, attn_bias):
        keys = queries if keys is None else keys
        values = keys if values is None else values
        
        if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
            print(queries.shape)
            print(keys.shape)
            print(values.shape)
            raise ValueError(
                "Inputs: quries, keys and values should all be 3-D tensors.")
        q, k, v = self.__compute_qkv(queries, keys, values)
        
        q = self.__split_heads(q, self.n_head)
        k = self.__split_heads(k, self.n_head)
        v = self.__split_heads(v, self.n_head)

        ctx_multiheads = self.compute_attention(q, k, v, attn_bias)
        out = self.__combine_heads(ctx_multiheads)

        # Project back to the model size.
        proj_out = self.output_linear(out)

        return proj_out


class positionwise_feed_forward(torch.nn.Module):
    """
    Position-wise Feed-Forward Layer.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    def __init__(
        self,
        d_model,
        d_ff,
        dropout_rate):
        super(positionwise_feed_forward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = F.gelu

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class encoder_layer(torch.nn.Module):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consists of a multi-head (self) attention sub-layer followed by
    a position-wise feed-forward sub-layer. Both two components are accompanied
    with the post_process_layer to add residual connection, layer normalization
    and dropout.
    """
    def __init__(
        self,
        n_head,
        d_key,
        d_value,
        d_model,
        prepostprocess_dropout,
        d_inner_hid,
        attention_dropout,
        relu_dropout,
        max_seq_len=11):
        super(encoder_layer, self).__init__()
        self.d_model = d_model
        self.prepostprocess_dropout = prepostprocess_dropout
        self.relu_dropout = relu_dropout
        self.ffn = positionwise_feed_forward(d_model=d_model, d_ff=d_inner_hid, dropout_rate=relu_dropout)
        self.attn_layer = multi_head_attention(
            d_key,
            d_value,
            d_model,
            n_head,
            attention_dropout)
        self.bn1 = nn.BatchNorm1d(max_seq_len, affine=True)
        self.bn2 = nn.BatchNorm1d(max_seq_len, affine=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(self.prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        attn_output = self.attn_layer(
                enc_input,
                None,
                None,
                attn_bias,)   
        attn_output = self.dropout(attn_output)
        attn_output += enc_input # skip connection

        #attn_output = self.bn1(attn_output)
        attn_output = self.ln1(attn_output)

        ffd_input = attn_output
        ffd_output = self.ffn(ffd_input)

        result = self.dropout(ffd_output)
        result += attn_output # skip connection

        #result = self.bn2(result)
        result = self.ln2(result)

        return result
        
        
class encoder(torch.nn.Module):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    def __init__(
        self,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        device,
        max_seq_len=11):
        super(encoder, self).__init__()
        self.layers = []#.to(torch.device("cuda:5"))
        self.n_layer = n_layer
        for i in range(n_layer):
            layer = encoder_layer(
                n_head=n_head,
                d_key=d_key,
                d_value=d_value,
                d_model=d_model,
                d_inner_hid=d_inner_hid,
                prepostprocess_dropout=prepostprocess_dropout,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                max_seq_len=max_seq_len).to(device)
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, enc_input, attn_bias):
        x = enc_input
        for layer in self.layers:
            x = layer(x, attn_bias)

        return x
