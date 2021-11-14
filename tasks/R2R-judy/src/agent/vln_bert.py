""" 
VLN-BERT model for scoring path-instruction pairs.
This implementation is based on https://github.com/arjunmajum/vln-bert
We made minor modifications to fit it in this code.
"""
import torch

from src.model import BertModel, BertPreTrainedModel, BertPreTrainingHeads


class VLNBert(BertPreTrainedModel):
    def __init__(self, config, dropout_prob=0.1):
        super().__init__(config)

        # vision and language processing streams
        self.bert = BertModel(config)

        # pre-training heads
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight
        )

        # path selection head
        self.vil_logit = torch.nn.Linear(config.bi_hidden_size, 1)

        # misc
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.fusion_method = config.fusion_method
        self.apply(self.init_bert_weights)

    def forward(
        self,
        instr_tokens,
        image_features,
        image_locations,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
    ):
        (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            _,
        ) = self.bert(
            input_txt=instr_tokens,
            input_imgs=image_features,
            image_loc=image_locations,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            image_attention_mask=image_attention_mask,
            co_attention_mask=co_attention_mask,
            output_all_encoded_layers=False,
        )

        linguisic_prediction, vision_prediction, _ = self.cls(
            sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
        )

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            assert False

        vil_logit = self.vil_logit(pooled_output)

        return (
            vil_logit,
            vision_prediction,
            linguisic_prediction,
        )
