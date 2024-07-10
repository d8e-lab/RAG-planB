# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Union, Optional
import torch.nn as nn
import torch
from transformers import BertModel
class Model(nn.Module):   
    def __init__(self, lm:Union[BertModel,str], pad_token_id, query_lstm:Optional[Union[nn.Module,str]]=None, corpus_lstm:Optional[Union[nn.Module,str]]=None, bidirectional=False, device="cuda:0",lm_freezed=True):
        super(Model, self).__init__()
        if isinstance(lm,str):
            self.lm = BertModel.from_pretrained(lm).to(device)
        else:
            self.lm = lm
        self.pad_token_id = pad_token_id
        self.num_directions = 2 if bidirectional else 1
        num_inputs = 768
        corpus_num_inputs = 6
        self.num_hiddens = 256

        """
        batch_first - If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False
        """
        if query_lstm is None:
            self.query_lstm_layer = nn.LSTM(num_inputs,self.num_hiddens,batch_first=True)
        else:
            if isinstance(query_lstm,nn.Module):
                self.query_lstm_layer = query_lstm
            else:
                self.query_lstm_layer = nn.LSTM(num_inputs,self.num_hiddens,batch_first=True)
                self.query_lstm_layer.load_state_dict(torch.load(query_lstm))
        self.query_lstm_layer.to(self.lm.device)

        if corpus_lstm is None:
            self.corpus_lstm_layer = nn.LSTM(num_inputs,self.num_hiddens,batch_first=True)
        else:
            if isinstance(corpus_lstm,nn.Module):
                self.corpus_lstm_layer = corpus_lstm
            else:
                self.corpus_lstm_layer = nn.LSTM(num_inputs,self.num_hiddens,batch_first=True)
                self.corpus_lstm_layer.load_state_dict(torch.load(corpus_lstm))
        self.corpus_lstm_layer.to(self.lm.device)
        
        self.query_state, self.corpus_state = self.begin_state(query_batch_size=1,corpus_batch_size=6,device=self.lm.device)

        if lm_freezed:
            for para in self.lm.parameters():
                para.requires_grad = False
    
    def forward(self, query_input_ids=None, corpus_intput_ids=None, attn_mask=None): 
        
        if query_input_ids is not None:
            attn_mask = query_input_ids.ne(self.pad_token_id) if attn_mask is None else attn_mask
            """lm_output (batch, seq, feature)"""
            lm_output = self.lm(input_ids=query_input_ids,attention_mask=attn_mask).last_hidden_state
            #TODO: mask padding
            _,last_state = self.query_lstm_layer(lm_output,self.query_state)
        if corpus_intput_ids is not None:
            attn_mask = corpus_intput_ids.ne(self.pad_token_id) if attn_mask is None else attn_mask
            """lm_output (batch, seq, feature)"""
            lm_output = self.lm(input_ids=corpus_intput_ids,attention_mask=attn_mask).last_hidden_state
            _,last_state = self.corpus_lstm_layer(lm_output,self.corpus_state)
        """last_state (h_n,c_n)
        h_n: tensor of shape (D * num_layers,H_out) for unbatched input or (D * num_layers, N, H_out) containing the final hidden state for each element in the sequence."""
        return last_state[0].transpose(0,1).squeeze(1)
    
    def save(self,save_path,tokenizer):
        query_bert_path=save_path/"bert"
        self.lm.save_pretrained(save_directory=query_bert_path)
        tokenizer.save_pretrained(query_bert_path)
        tokenizer.save_vocabulary(query_bert_path)
        torch.save(self.query_lstm_layer.state_dict(),save_path/'query_lstm.pt')
        torch.save(self.query_lstm_layer.state_dict(),save_path/'corpus_lstm.pt')
    
    def save_(self, output_dir: str,tokenizer):
        import os
        os.makedirs(output_dir,exist_ok=True)
        state_dict = self.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,v in state_dict.items()})
        # self.model.save_pretrained(output_dir, state_dict=state_dict)
        torch.save(state_dict,output_dir/"torch.pt")
        tokenizer.save_pretrained(output_dir)
        tokenizer.save_vocabulary(output_dir)

    def begin_state(self, device, query_batch_size=1,corpus_batch_size=5):
        if not isinstance(self.query_lstm_layer, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return  torch.zeros((self.num_directions * self.query_lstm_layer.num_layers,
                                query_batch_size, self.num_hiddens),
                                device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((
                        self.num_directions * self.query_lstm_layer.num_layers,
                        query_batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.query_lstm_layer.num_layers,
                        query_batch_size, self.num_hiddens), device=device)),(
                    torch.zeros((
                        self.num_directions * self.corpus_lstm_layer.num_layers,
                        corpus_batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.corpus_lstm_layer.num_layers,
                        corpus_batch_size, self.num_hiddens), device=device))
                
      
        
 
