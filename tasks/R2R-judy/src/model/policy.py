import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
logger = logging.getLogger("main.policy")

from . import units as U


# ---------------------------
# --   Speaker-Follower    --
# ---------------------------

class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention 
        over instructions for decoding navigation actions. '''

    def __init__(self, 
        hidden_size, drop_ratio, action_embed_size:int=2048+128,
        feature_size:int=2048+128, image_attn_layers=None
    ):
        super(AttnDecoderLSTM, self).__init__()

        self.action_embed_size = action_embed_size  # 2048 + 128
        self.feature_size = feature_size            # 2048 + 128
        self.hidden_size = hidden_size

        self.drop = nn.Dropout(p=drop_ratio)
        self.lstm = nn.LSTMCell(action_embed_size + feature_size, hidden_size)
        self.text_attn = U.SoftDotAttention(hidden_size)
        self.visual_attn = U.VisualSoftDotAttention(hidden_size, feature_size)
        
        # decode to action choice
        self.decode_action = U.ActionScoring(action_embed_size, hidden_size)
    
    def forward(self, img_feature, a_t_prev, a_t_cands, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).
        
        :param img_feature: batch x v_dim(36) x feature_size
        :param a_t_prev:    batch x action_embed_size
        :param a_t_cands:   batch x num_candidates x action_embed_size
        :param h_0:         batch x hidden_size
        :param c_0:         batch x hidden_size
        :param ctx:         batch x seq_len x dim, context
        :param ctx_mask:    batch x seq_len - indices to be masked
        '''
        weighted_v, alpha_v = \
            self.visual_attn(h_0, img_feature)  # (batch, feature_size), (batch, v_dim)
        visual_ctx = self.drop(torch.cat(
            (a_t_prev, weighted_v), dim=1))     # (batch, action_embed_size+feature_size)
        
        h_1, c_1 = self.lstm(visual_ctx, (h_0, c_0))  # (batch, hidden_size), (batch, hidden_size)
        h_1_drop = self.drop(h_1)
        
        h_tilde, alpha_c = \
            self.text_attn(h_1_drop, ctx, ctx_mask)  # (batch, hidden_size), (batch, seq_len)
        a_t_logit = self.decode_action(a_t_cands, h_tilde)  # (batch, num_candidates)

        return a_t_logit, (h_1, c_1), (alpha_c, alpha_v)


# ---------------------------
# --   Self-Monitoring     --
# ---------------------------

class MonitorDecoder(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(self, 
        rnn_hidden_size:int, drop_ratio:float, 
        max_enc_len:int, mlp_dims:list=[128, 1024], 
        action_embed_size:int=2048+128, feature_size:int=2048+128
    ):
        super(MonitorDecoder, self).__init__()

        self.rnn_hidden_size = rnn_hidden_size
        self.max_enc_len = max_enc_len
        self.mlp_dims = mlp_dims
        self.feature_size = feature_size
        self.action_embed_size = action_embed_size
        self.img_hidden_size = mlp_dims[-1]

        self.proj_navigable_mlp = U.MLPwithBN(
            input_size=self.action_embed_size, hidden_size=mlp_dims, 
            use_bn=True, dropout=0.5, use_bias=True, relu=True)
        
        self.position = U.PositionalEncoding(
            rnn_hidden_size, dropout=0.1, max_len=max_enc_len)
        
        self.text_attn = U.SoftDotAttention(rnn_hidden_size, context_only=True)
        self.visual_attn = U.VisualSoftDotAttention(
            rnn_hidden_size, None, self.img_hidden_size)

        self.drop = nn.Dropout(p=drop_ratio)
        self.lstm = nn.LSTMCell(
            self.img_hidden_size*2 + rnn_hidden_size, rnn_hidden_size)
        
        # decode to action choice
        self.action_linear = nn.Linear(rnn_hidden_size * 2, self.img_hidden_size)

        # progress monitor
        self.monitor_linear = nn.Linear(\
            rnn_hidden_size + self.img_hidden_size, rnn_hidden_size, bias=True)
        self.critic = nn.Sequential( nn.Linear(max_enc_len + rnn_hidden_size, 1), 
                                     nn.Tanh())

    def policy_net(self, weighted_ctx, hidden, cands_rep):
        ''' Decode action representations into logits 
        
        :param weighted_ctx: context, batch x rnn_hidden_size
        :param hidden: decoder lstm hidden layer output, batch x rnn_hidden_size
        :param cands_rep: candidates representation, batch x num_candidates x img_hidden_size
        '''
        h_tilde = self.action_linear(torch.cat((weighted_ctx, hidden), dim=1))      # batch x img_hidden_size
        logit = torch.bmm(cands_rep, h_tilde.unsqueeze(2)).squeeze(2)               # batch x num_candidates
        return logit
    
    def progress_monitor(self, h_0, c_1, weighted_cands, ctx_attn):
        ''' A progress monitor that serves as regularizer during training 
            and prunes unfinished trajectories during inference. 
        :param h_0:              batch x rnn_hidden_size
        :param c_1:              batch x rnn_hidden_size
        :param weighted_cands:   batch x img_hidden_size
        :param ctx_atten:        batch x max_enc_len
        '''
        concat_input = self.monitor_linear(torch.cat((h_0, weighted_cands), dim=1)) # batch x rnn_hidden_size
        h_pm = self.drop(torch.sigmoid(concat_input)*torch.tanh(c_1))               # batch x rnn_hidden_size
        progress_value = self.critic(torch.cat((ctx_attn, h_pm), dim=1))            # batch x 1
        return progress_value.squeeze()                                             # batch

    def forward(self, img_feature, a_t_prev, a_t_cands, h_0, c_0, ctx, ctx_mask=None, candidate_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).
        
        :param img_feature:     batch x v_dim(36) x feature_size, not used in this module
        :param a_t_prev:        batch x action_embed_size
        :param a_t_cands:       batch x num_candidates x action_embed_size
        :param h_0:             batch x rnn_hidden_size
        :param c_0:             batch x rnn_hidden_size
        :param ctx:             batch x seq_len x rnn_hidden_size, context
        :param ctx_mask:        batch x seq_len - indices to be masked
        :param candidate_mask:  batch x num_candidates, True is needed to be masked
        '''
        proj_a_t_prev = self.proj_navigable_mlp(a_t_prev)                           # batch x img_hidden_size
        batch_size, num_candidates, _ = a_t_cands.shape
        proj_a_t_cands = self.proj_navigable_mlp(a_t_cands\
            .view(-1, self.action_embed_size)).view(batch_size, num_candidates, -1) # batch x num_candidates x img_hidden_size
        proj_a_t_cands = proj_a_t_cands * \
            (1-candidate_mask.float()).unsqueeze(2).expand_as(proj_a_t_cands)

        positioned_ctx = self.position(ctx)                                         # batch x seq_len x rnn_hidden_size
        weighted_ctx, ctx_attn = self.text_attn(h_0, positioned_ctx, ctx_mask)      # batch x rnn_hidden_size
        weighted_cands, cands_v_attn = \
            self.visual_attn(h_0, proj_a_t_cands, candidate_mask)                   # batch x img_hidden_size
        
        # merge info into one LSTM to be carry through time
        concat_input = torch.cat((proj_a_t_prev, weighted_cands, weighted_ctx), 1)
        h_1, c_1 = self.lstm(concat_input, (h_0, c_0))                              # batch x rnn_hidden_size
        
        # compute action logits
        a_t_logit = self.policy_net(weighted_ctx, self.drop(h_1), proj_a_t_cands)

        # compute current progress
        cur_progress = self.progress_monitor(h_0, c_1, weighted_cands, ctx_attn)

        return (a_t_logit, cur_progress), (h_1, c_1), (ctx_attn, cands_v_attn)


# ---------------------------
# -- Environmental Dropout --
# ---------------------------

class EnvDropDecoder(nn.Module):
    ''' Learning to Navigate Unseen Environments: Back Translation with Environmental Dropout '''
    
    def __init__(self, hidden_size, drop_ratio, feat_drop_ratio,
        action_embed_size:int=64, angle_feat_size:int=128, feature_size:int=2048+128, 
    ):
        super(EnvDropDecoder, self).__init__()

        self.feature_size = feature_size            # 2048 + 128
        self.action_embed_size = action_embed_size  # 64
        self.angle_feat_size = angle_feat_size      # 128
        self.hidden_size = hidden_size

        self.act_embed = nn.Sequential(
            nn.Linear(self.angle_feat_size, self.action_embed_size), 
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=drop_ratio)
        self.env_drop = nn.Dropout(p=feat_drop_ratio)
        self.lstm = nn.LSTMCell(action_embed_size + feature_size, hidden_size)
        # attention layers
        self.text_attn = U.SoftDotAttention(hidden_size)
        self.visual_attn = U.SoftDotAttention(
            hidden_size, context_dim=self.feature_size, context_only=True)
        self.cand_attn = nn.Linear(hidden_size, self.feature_size, bias=False)
    
    def candidate_attn(self, h_tilde_drop, cand_feat):
        """ 
        :param h_tilde_drop: batch x hidden_size
        :param cand_feat:    batch x num_candidates x feature_size
        """
        target = self.cand_attn(h_tilde_drop).unsqueeze(2)      # batch x feature_size x 1
        logit = torch.bmm(cand_feat, target).squeeze(2)         # batch x num_candidates
        return logit
    
    def forward(self, 
        a_t_prev, img_feature, cand_feature, h_tilde_prev, 
        h_0, c_0, ctx, ctx_mask=None, already_dropfeat=False
    ):
        ''' Takes a single step in the decoder LSTM (allowing sampling).
        
        :param a_t_prev:        batch x action_embed_size
        :param img_feature:     batch x v_dim(36) x feature_size
        :param cand_feat:       batch x num_candidates x feature_size
        :param h_tilde_prev:    batch x hidden_size
        :param h_0:             batch x hidden_size
        :param c_0:             batch x hidden_size
        :param ctx:             batch x seq_len x dim, context
        :param ctx_mask:        batch x seq_len - indices to be masked
        :param already_dropfeat: used in EnvDrop
        '''
        prev_act_emb = self.drop(self.act_embed(a_t_prev))

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            img_feature[..., :-self.angle_feat_size] = \
                self.env_drop(img_feature[..., :-self.angle_feat_size]) 
            cand_feature[..., :-self.angle_feat_size] = \
                self.env_drop(cand_feature[..., :-self.angle_feat_size])
            # Do not drop the last args.angle_feat_size (position feat)
        
        prev_h1_drop = self.drop(h_tilde_prev)
        visual_feat, alpha_v = self.visual_attn(prev_h1_drop, img_feature)

        concat_input = torch.cat((prev_act_emb, visual_feat), 1)    # (batch, action_embed_size_size + feature_size)
        h_1, c_1 = self.lstm(concat_input, (h_tilde_prev, c_0))     # (batch, hidden_size) *2

        h_1_drop = self.drop(h_1)
        h_tilde, alpha_c = self.text_attn(h_1_drop, ctx, ctx_mask)  # (batch, hidden_size), (batch, seq_len)

        h_tilde_drop = self.drop(h_tilde)
        logit = self.candidate_attn(h_tilde_drop, cand_feature)     # (batch, num_candidates)

        return logit, (h_1, c_1), h_tilde


class Critic(nn.Module):
    def __init__(self, hidden_size, drop_ratio):
        super(Critic, self).__init__()

        self.hidden_size = hidden_size
        self.drop_ratio = drop_ratio

        self.state2value = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.drop_ratio),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, state):
        """ 
        :param state: batch x hidden_size
        """
        return self.state2value(state).squeeze()


# ---------------------------
# --    Sub-Instruction    --
# ---------------------------

class SubMonitorDecoder(nn.Module):
    ''' Sub-Instruction Aware VLN based on Self-Monitoring. '''

    def __init__(self, 
        rnn_hidden_size:int, drop_ratio:float, 
        max_enc_len:int, mlp_dims:list=[128, 1024], 
        action_embed_size:int=2048+128, feature_size:int=2048+128
    ):
        super(SubMonitorDecoder, self).__init__()

        self.rnn_hidden_size = rnn_hidden_size
        self.max_enc_len = max_enc_len
        self.mlp_dims = mlp_dims
        self.feature_size = feature_size
        self.action_embed_size = action_embed_size
        self.img_hidden_size = mlp_dims[-1]

        self.proj_navigable_mlp = U.MLPwithBN(
            input_size=self.action_embed_size, hidden_size=mlp_dims, 
            use_bn=True, dropout=0.5, use_bias=True, relu=True)

        self.position = U.PositionalEncoding(
            rnn_hidden_size, dropout=0.1, max_len=max_enc_len)
        
        self.text_attn = U.SoftDotBlockAttention(rnn_hidden_size)
        self.visual_attn = U.VisualSoftDotAttention(
            rnn_hidden_size, None, self.img_hidden_size)

        self.drop = nn.Dropout(p=drop_ratio)
        self.lstm = nn.LSTMCell(
            self.img_hidden_size*2 + rnn_hidden_size, rnn_hidden_size)
        
        # decode to action choice
        self.action_linear = nn.Linear(rnn_hidden_size * 2, self.img_hidden_size)

    def policy_net(self, weighted_ctx, hidden, cands_rep):
        ''' Decode action representations into logits 
        
        :param weighted_ctx: context, batch x rnn_hidden_size
        :param hidden: decoder lstm hidden layer output, batch x rnn_hidden_size
        :param cands_rep: candidates representation, batch x num_candidates x img_hidden_size
        '''
        h_tilde = self.action_linear(torch.cat((weighted_ctx, hidden), dim=1))      # batch x img_hidden_size
        logit = torch.bmm(cands_rep, h_tilde.unsqueeze(2)).squeeze(2)               # batch x num_candidates
        return logit
    
    def forward(self, 
        img_feature, a_t_prev, a_t_cands, h_0, c_0, ctx, 
        sub_seq_lengths:np.ndarray, selected_block_idx:np.ndarray, 
        ctx_mask=None, candidate_mask=None
    ):
        ''' Takes a single step in the decoder LSTM (allowing sampling).
        
        :param img_feature:     batch x v_dim(36) x feature_size, not used in this module
        :param a_t_prev:        batch x action_embed_size
        :param a_t_cands:       batch x num_candidates x action_embed_size
        :param h_0:             batch x rnn_hidden_size
        :param c_0:             batch x rnn_hidden_size
        :param ctx:             batch x seq_len x rnn_hidden_size, context
        :sub_seq_lengths:       batch x max_subinstr_size, lengths of sub-instructions
        :selected_block_idx:    batch, current sub-instruction index
        :param ctx_mask:        batch x seq_len - indices to be masked
        :param candidate_mask:  batch x num_candidates
        '''
        proj_a_t_prev = self.proj_navigable_mlp(a_t_prev)                           # batch x img_hidden_size
        batch_size, num_candidates, _ = a_t_cands.shape
        proj_a_t_cands = self.proj_navigable_mlp(a_t_cands\
            .view(-1, self.action_embed_size)).view(batch_size, num_candidates, -1) # batch x num_candidates x img_hidden_size
        proj_a_t_cands = proj_a_t_cands * \
            (1-candidate_mask.float()).unsqueeze(2).expand_as(proj_a_t_cands)

        positioned_ctx = self.position(ctx)                                         # batch x seq_len x rnn_hidden_size
        weighted_ctx, ctx_attn = self.text_attn(
            h_0, positioned_ctx, sub_seq_lengths, selected_block_idx, ctx_mask)     # batch x rnn_hidden_size
        weighted_cands, cands_v_attn = self.visual_attn(\
            h_0, proj_a_t_cands, candidate_mask)                                    # batch x img_hidden_size
        
        # merge info into one LSTM to be carry through time
        concat_input = torch.cat((proj_a_t_prev, weighted_cands, weighted_ctx), 1)
        h_1, c_1 = self.lstm(concat_input, (h_0, c_0))                              # batch x rnn_hidden_size
        
        # compute action logits
        a_t_logit = self.policy_net(weighted_ctx, self.drop(h_1), proj_a_t_cands)

        return a_t_logit, (h_1, c_1), weighted_ctx, (ctx_attn, cands_v_attn)


class InstrShifting(nn.Module):
    ''' Sub-Instruction Shifting Module.
        Decide whether the current subinstruction will 
        be completed by the next action or not. '''
    
    def __init__(self, rnn_hidden_size, shift_hidden_size, action_emb_size, max_subinstr_size, drop_ratio):
        super(InstrShifting, self).__init__()

        self.drop = nn.Dropout(p=drop_ratio)
        self.linear0 = nn.Linear(rnn_hidden_size, shift_hidden_size, bias=False)
        self.linear1 = nn.Linear(rnn_hidden_size + shift_hidden_size + action_emb_size, 
                                 shift_hidden_size, bias=False)
        self.linear2 = nn.Linear(max_subinstr_size, shift_hidden_size, bias=False)
        self.linear3 = nn.Linear(2*shift_hidden_size, 1, bias=False)
    
    def forward(self, h_t, m_t, a_t_cur, weighted_ctx, e_t):
        ''' Propogate through the network. 
        :param h_t:          torch.Tensor, batch x rnn_hidden_size
        :param m_t:          torch.Tensor, batch x rnn_hidden_size
        :param a_t_cur:      torch.Tensor, batch x action_emb_size
        :param weighted_ctx: torch.Tensor, batch x rnn_hidden_size
        :param e_t:          torch.Tensor, batch x max_subinstr_size
        '''
        proj_h = self.linear0(self.drop(h_t))                               # batch x shift_hidden_size
        concat_input = torch.cat((proj_h, a_t_cur, weighted_ctx), 1)        # batch x (shift_hidden_size + rnn_hidden_size + action_emb_size)
        h_t_c = torch.sigmoid(self.linear1(concat_input))*torch.tanh(m_t)   # batch x shift_hidden_size

        proj_e = self.linear2(e_t)                                          # batch x shift_hidden_size
        concat_input = torch.cat((proj_e, self.drop(h_t_c)), 1)             # batch x shift_hidden_size*2
        p_t_s = torch.sigmoid(self.linear3(concat_input))                   # batch x 1

        return p_t_s.squeeze()                                              # batch x 1
    
