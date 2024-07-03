# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch
from transformers import BertModel
class Model(nn.Module):   
    def __init__(self, encoder:BertModel,pad_token_id):
        super(Model, self).__init__()
        self.encoder = encoder
        self.pad_token_id = pad_token_id
      
    def forward(self, query_input_ids=None, attn_mask=None): 
        # if code_inputs is not None:
        #     nodes_mask=position_idx.eq(0)
        #     token_mask=position_idx.ge(2)        
        #     inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
        #     nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        #     nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        #     avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        #     inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        #     return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1]
        # else:
        attn_mask = query_input_ids.ne(self.pad_token_id) if attn_mask is None else attn_mask
        output = self.encoder(input_ids=query_input_ids,attention_mask=attn_mask)
        return output[1]

      
        
 
