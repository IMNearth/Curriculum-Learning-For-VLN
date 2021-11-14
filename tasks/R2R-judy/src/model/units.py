import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import math
import logging
logger = logging.getLogger("main.units")


class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''
    
    def __init__(self, 
        vocab_size, embed_size, hidden_size, padding_idx, 
        drop_ratio=0.5, bidirectional=False, num_layers=1, glove=None
    ):
        super(EncoderLSTM, self).__init__()

        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers
        self.use_glove = (glove is not None)

        self.drop = nn.Dropout(p=drop_ratio)
        if self.use_glove:
            print("\t Use GloVe embedding. ")
            logger.info("\t Use GloVe embedding. ")
            self.embedding = \
                nn.Embedding.from_pretrained(torch.from_numpy(glove), freeze=True)
        else:
            self.embedding = \
                nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        
        self.lstm = nn.LSTM(self.embed_size, 
                            self.hidden_size, 
                            dropout=(drop_ratio)*(num_layers>1), 
                            num_layers=self.num_layers, 
                            batch_first=True, 
                            bidirectional=bidirectional)
        self.enc2dec = nn.Linear(self.hidden_size*self.num_directions, 
                                 self.hidden_size*self.num_directions)
        
    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor, already_sorted:bool=True):
        '''
        :param inputs: torch.Tensor(batch_size, max_seq_len)
        :param lengths: torch.Tensor(batch_size, )
        '''
        max_seq_len = inputs.shape[1]
        embeds = self.embedding(inputs)                                     # [batch, max_len, embed_size]
        if not self.use_glove:
            embeds = self.drop(embeds)
        
        packed_in = pack_padded_sequence(embeds, lengths, \
            batch_first=True, enforce_sorted=already_sorted)                # by default, enforce_sorted = True
        packed_out, (enc_h_t, enc_c_t) = self.lstm(packed_in)               # (h_0, c_0) both default to zero
        # enc_h_t, enc_c_t: torch.Tensor(num_layers * num_directions, batch, hidden_size)

        if self.num_directions == 2: # bidirectional
            h_t = torch.cat((enc_h_t[-2], enc_h_t[-1]), 1)                  # [-2] is forward, [-1] is backward
            c_t = torch.cat((enc_c_t[-2], enc_c_t[-1]), 1)                  # [batch, hidden_size * 2]
        else:
            h_t, c_t = enc_h_t[-1], enc_c_t[-1]                             # [batch, hidden_size]
        
        decoder_init = nn.Tanh()(self.enc2dec(h_t))

        ctx, lengths = pad_packed_sequence(packed_out, batch_first=True, total_length=max_seq_len)
        ctx = self.drop(ctx)                                                # [batch, max_len, hidden_size*num_directions]

        return ctx, decoder_init, c_t


class SoftDotAttention(nn.Module):
    ''' Soft Dot Attention.
        Ref: http://www.aclweb.org/anthology/D15-1166
        Adapted from PyTorch OPEN NMT. '''

    def __init__(self, query_dim, context_only=False, context_dim=None):
        ''' Initialize layer. 
            If Context_Only is True, like SoftDot, 
            but don't concatenat h or perform the non-linearity transform '''
        super(SoftDotAttention, self).__init__()

        self.context_only = context_only
        
        ctx_dim = query_dim if context_dim is None else context_dim
        if context_only:
            self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        else:
            self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
            self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
            self.tanh = nn.Tanh()
        
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        :param h:          batch x query_dim
        :param context:    batch x seq_len x ctx_dim
        :param mask:       batch x seq_len, indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if self.context_only: return weighted_context, attn

        h_tilde = torch.cat((weighted_context, h), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))  # batch x dim
        return h_tilde, attn


class VisualSoftDotAttention(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, h_dim:int, v_dim:int=None, dot_dim:int=256):
        '''Initialize layer.'''
        super(VisualSoftDotAttention, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        
        self.use_v_linear = (v_dim is not None)
        if self.use_v_linear:
            self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, visual_context, mask=None):
        '''Propagate h through the network.
        
        :param h: batch x h_dim
        :param visual_context: batch x v_num x v_dim
        '''
        target = self.linear_in_h(h).unsqueeze(2)           # batch x dot_dim x 1

        if self.use_v_linear: context = self.linear_in_v(visual_context)
        else: context = visual_context                      # batch x v_num x dot_dim
        assert context.shape[-1] == target.shape[-2]

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)        # batch x v_num
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))    # batch x 1 x v_num

        weighted_context = torch.bmm(
            attn3, visual_context).squeeze(1)               # batch x v_dim
        return weighted_context, attn


class ActionScoring(nn.Module):
    ''' Linearly mapping h and v to the same dimension, 
        and do a elementwise multiplication and a linear scoring. '''
    
    def __init__(self, action_size, hidden_size, dot_size:int=256):
        super(ActionScoring, self).__init__()

        self.linear_act = nn.Linear(action_size, dot_size, bias=True)
        self.linear_hid = nn.Linear(hidden_size, dot_size, bias=True)
        self.linear_out = nn.Linear(dot_size, 1, bias=True)
    
    def forward(self, act_cands, h_tilde):
        ''' Compute logits of action candidates
        :param act_cands: torch.Tensor(batch, num_candidates, action_emb_size)
        :param h_tilde: torch.Tensor(batch, hidden_size)
        
        Return -> torch.Tensor(batch, num_candidates)
        '''
        target = self.linear_hid(h_tilde).unsqueeze(1)  # (batch, 1, dot_size)
        context = self.linear_act(act_cands)    # (batch, num_cands, dot_size)
        product = torch.mul(context, target)    # (batch, num_cands, dot_size)
        logits = self.linear_out(product).squeeze(2)      # (batch, num_cands)
        return logits


class PositionalEncoding(nn.Module):
    """Implement the PE function to introduce the concept of relative position"""

    def __init__(self, d_model, dropout, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i + 1
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class MLPwithBN(nn.Module):
    ''' Adapted from "build_mlp" function in 
        https://github.com/chihyaoma/selfmonitoring-agent/ ''' 
    
    def __init__(self, 
        input_size:int, hidden_size:tuple, out_size:int=None,
        dropout:float=.0, use_bn:bool=False, use_bias:bool=True, relu=True):
        super(MLPwithBN, self).__init__()

        self.in_size = input_size

        layers = []
        if use_bn: layers.append(nn.BatchNorm1d(input_size))

        dim_list = [input_size] + list(hidden_size)
        for i in range(len(dim_list)-1):
            d_in, d_out = dim_list[i], dim_list[i+1]
            layers.append(nn.Linear(d_in, d_out, bias=use_bias))
            if use_bn: layers.append(nn.BatchNorm1d(d_out))
            if dropout > 0: layers.append(nn.Dropout(p=dropout))
            if relu: layers.append(nn.ReLU(inplace=True))
        
        self.out_size = hidden_size[-1]
        
        if out_size: 
            layers.append(nn.Linear(dim_list[-1], out_size, bias=use_bias))
            self.out_size = out_size
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """ x: torch.Tensor(batch, in_size) """
        return self.mlp(x)


class SoftDotBlockAttention(nn.Module):
    ''' Soft Dot Attention For sub-instructions. '''

    def __init__(self, dim):
        ''' Initialize layer. '''
        super(SoftDotBlockAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, h, context, sub_seq_lengths, selected_block_idx, mask=None):
        '''Propagate h through the network.

        :param h:          batch x dim
        :param context:    batch x seq_len x dim
        :param mask:       batch x seq_len, indices to be masked
        '''
        batch, dim = h.shape
        device = next(self.parameters()).device
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))

        # compute weighted ctx for each block
        weighted_context = torch.zeros(batch, dim).to(device)
        for idx in range(batch):
            sub_seq_len = sub_seq_lengths[idx]
            
            ''' re attend the step-selected block '''
            max_c_dx = selected_block_idx[idx]
            sc = sum(sub_seq_len[0:max_c_dx]) + 1
            block_softattn = self.softmax(attn[idx, sc:sc+sub_seq_len[max_c_dx]])
            block_value = context[idx, sc:sc+sub_seq_len[max_c_dx]]
            weighted_context[idx] = torch.matmul(block_softattn, block_value)

        return weighted_context, attn


class SpeakerEncoder(nn.Module):
    """ 
    source: https://github.com/airsplay/R2R-EnvDrop
    path: r2r_src/model.py#207
    """
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional, angle_feat_size, feat_dropout):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size
        self.angle_feat_size = angle_feat_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=feat_dropout)
        self.attention_layer = SoftDotAttention(query_dim=self.hidden_size, context_dim=feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-self.angle_feat_size] = self.drop3(x[..., :-self.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-self.angle_feat_size] = self.drop3(feature[..., :-self.angle_feat_size])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x


class SpeakerDecoder(nn.Module):
    """ 
    source: https://github.com/airsplay/R2R-EnvDrop
    path: r2r_src/model.py#259
    """
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(query_dim=hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


