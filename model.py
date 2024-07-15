# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Union, Optional
import torch.nn as nn
import torch
from transformers import BertModel
class Model(nn.Module):   
    def __init__(self, lm:Union[BertModel,str], pad_token_id, query_lstm:Optional[Union[nn.Module,str]]=None, corpus_lstm:Optional[Union[nn.Module,str]]=None, corpus_batch_size = 6, bidirectional=False, device="cuda:0", lm_freezed=True,embedding_type="query"):
        super(Model, self).__init__()
        if isinstance(lm,str):
            self.lm = BertModel.from_pretrained(lm).to(device)
        else:
            self.lm = lm
        self.pad_token_id = pad_token_id
        num_inputs = 768 #len(bert.vocab) = 768
        corpus_num_inputs = 6
        self.num_hiddens = 256
        self.embedding_type = embedding_type
        """
        batch_first - If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False
        """
        if query_lstm is None:
            self.query_lstm_layer = Encoder(num_inputs, self.num_hiddens, batch_first=True, bidirectional = bidirectional,device = self.lm.device)
        else:
            if isinstance(query_lstm,nn.Module):
                self.query_lstm_layer = query_lstm.to(self.lm.device)
            else:
                self.query_lstm_layer = Encoder(num_inputs, self.num_hiddens, batch_first=True, bidirectional = bidirectional,  device = self.lm.device)
                self.query_lstm_layer.load_state_dict(torch.load(query_lstm))
        self.query_lstm_layer.to(self.lm.device)

        if corpus_lstm is None:
            self.corpus_lstm_layer = Encoder(num_inputs, self.num_hiddens, batch_first=True, bidirectional = bidirectional, device = self.lm.device)
        else:
            if isinstance(corpus_lstm,nn.Module):
                self.corpus_lstm_layer = corpus_lstm.to(self.lm.device)
            else:
                self.corpus_lstm_layer = Encoder(num_inputs, self.num_hiddens, batch_first=True, bidirectional = bidirectional, device = self.lm.device)
                self.corpus_lstm_layer.load_state_dict(torch.load(corpus_lstm))
        self.corpus_lstm_layer.to(self.lm.device)

        if lm_freezed:
            for para in self.lm.parameters():
                para.requires_grad = False
        else:
            print("-------lm is not freezed!--------")
    
    def forward(self, input_ids=None, attn_mask=None): 
        
        if self.embedding_type=="query":
            attn_mask = input_ids.ne(self.pad_token_id) if attn_mask is None else attn_mask
            """lm_output (batch, seq, feature)"""
            lm_output = self.lm(input_ids=input_ids,attention_mask=attn_mask).last_hidden_state
            #TODO: mask padding
            self.mask_pad_token_(lm_output,attn_mask,self.pad_token_id)
            encoding = self.query_lstm_layer(lm_output)
        if self.embedding_type=="corpus":
            attn_mask = input_ids.ne(self.pad_token_id) if attn_mask is None else attn_mask
            """lm_output (batch, seq, feature)"""
            lm_output = self.lm(input_ids=input_ids,attention_mask=attn_mask).last_hidden_state
            self.mask_pad_token_(lm_output,attn_mask,self.pad_token_id)
            encoding = self.corpus_lstm_layer(lm_output)

        return encoding
    
    def set_query_mode(self):
        self.embedding_type = "query"
    
    def set_corpus_mode(self):
        self.embedding_type = "corpus"
    
    def mask_pad_token_(self,lm_output,attn_mask,mask_value):
        """To mask pad token in the output of bert"""
        attn_mask = attn_mask.unsqueeze(-1).bool()
        return lm_output.masked_fill_(mask=attn_mask,value=mask_value)

    def save(self,save_path,tokenizer):
        query_bert_path=save_path/"bert"
        self.lm.save_pretrained(save_directory=query_bert_path)
        tokenizer.save_pretrained(query_bert_path)
        tokenizer.save_vocabulary(query_bert_path)
        torch.save(self.query_lstm_layer.state_dict(),save_path/'query_encoder.pt')
        torch.save(self.corpus_lstm_layer.state_dict(),save_path/'corpus_encoders.pt')
    
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

    # 弃用
    @DeprecationWarning
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
                
class Encoder(nn.Module):
    def __init__(self, num_inputs, num_hiddens, batch_first = True, bidirectional = False, device = "cuda:0", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_hiddens = num_hiddens
        self.num_directions = 2 if bidirectional else 1
        self.device = device
        # TODO: mlp & relu etc.
        self.lstm_layter = nn.LSTM(num_inputs,self.num_hiddens,batch_first=batch_first)

    def forward(self,input):
        """last_state (h_n,c_n)
        h_n: tensor of shape (D * num_layers,H_out) for unbatched input or (D * num_layers, N, H_out) containing the final hidden state for each element in the sequence."""
        batch_size = input.size(0)  # 从输入读取batch_size
        self.state = self.begin_state(batch_size = batch_size, device=self.device)
        _,last_state = self.lstm_layter(input,self.state)
        if self.training:
            return last_state[0].transpose(0,1).squeeze(1)
        else:
            return (last_state[0].transpose(0,1).repeat(1,input.shape[1],1),last_state[1])

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.lstm_layter, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return  torch.zeros((self.num_directions * self.lstm_layter.num_layers,
                                batch_size, self.num_hiddens),
                                device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((
                        self.num_directions * self.lstm_layter.num_layers,
                        batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.lstm_layter.num_layers,
                        batch_size, self.num_hiddens), device=device))

