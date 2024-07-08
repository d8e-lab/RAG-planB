# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch
from transformers import BertModel
from FlagEmbedding import BGEM3FlagModel
class Model(nn.Module):   
    def __init__(self, query_encoder,corpus_encoder,pad_token_id=None):
        super(Model, self).__init__()
        self.query_encoder = query_encoder
        self.corpus_encoder = corpus_encoder
        self.pad_token_id = pad_token_id
        
    def forward(self, query_input_ids=None, corpus_intput_ids=None, attn_mask=None): 
        
        if query_input_ids is not None:
            output = self.query_encoder.encode(query_input_ids)['dense_vecs']
        if corpus_intput_ids is not None:
            output = self.corpus_encoder.encode(corpus_intput_ids)['dense_vecs']
        return output
    def parameters(self):
        return [i for i in self.query_encoder.model.parameters()] + [i for i in self.corpus_encoder.model.parameters()]
    def save(self,save_path,tokenizer=None):
        query_encoder_path=save_path/"query_bert"
        corpus_encoder_path=save_path/"corpus_bert"
        self.query_encoder.model.save(output_dir=query_encoder_path)
        self.corpus_encoder.model.save(output_dir=corpus_encoder_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(query_encoder_path)
            tokenizer.save_vocabulary(query_encoder_path)
            tokenizer.save_pretrained(corpus_encoder_path)
            tokenizer.save_vocabulary(corpus_encoder_path)
      
        
 
