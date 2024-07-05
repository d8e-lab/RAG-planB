# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch
from transformers import BertModel
class Model(nn.Module):   
    def __init__(self, query_encoder:BertModel,corpus_encoder:BertModel,pad_token_id):
        super(Model, self).__init__()
        self.query_encoder = query_encoder
        self.corpus_encoder = corpus_encoder
        self.pad_token_id = pad_token_id
        
    def forward(self, query_input_ids=None, corpus_intput_ids=None, attn_mask=None): 
        
        if query_input_ids is not None:
            attn_mask = query_input_ids.ne(self.pad_token_id) if attn_mask is None else attn_mask
            output = self.query_encoder(input_ids=query_input_ids,attention_mask=attn_mask)
        if corpus_intput_ids is not None:
            attn_mask = corpus_intput_ids.ne(self.pad_token_id) if attn_mask is None else attn_mask
            output = self.corpus_encoder(input_ids=corpus_intput_ids,attention_mask=attn_mask)
        return output[1]
    
    def save(self,save_path,tokenizer):
        query_bert_path=save_path/"query_bert"
        corpus_bert_path=save_path/"corpus_bert"
        self.query_encoder.save_pretrained(save_directory=query_bert_path)
        self.corpus_encoder.save_pretrained(save_directory=corpus_bert_path)
        tokenizer.save_pretrained(query_bert_path)
        tokenizer.save_vocabulary(query_bert_path)
        tokenizer.save_pretrained(corpus_bert_path)
        tokenizer.save_vocabulary(corpus_bert_path)
      
        
 
