# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Union, Optional
import torch.nn as nn
import torch
from transformers import BertModel
class Model(nn.Module):   
    def __init__(self, lm:Union[BertModel,str], pad_token_id, query_lstm:Optional[Union[nn.Module,str]]=None, corpus_lstm:Optional[Union[nn.Module,str]]=None, corpus_batch_size = 6, bidirectional=False, device="cuda:0", lm_freezed=True,embedding_type="query",lstm_num_layers:int = 1, normalized:bool = False):
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
        """load dual encoders."""
        if query_lstm is None:
            self.query_lstm_layer = Encoder(num_inputs, self.num_hiddens, batch_first=True, bidirectional = bidirectional,device = self.lm.device,num_layers=lstm_num_layers,normalized=normalized)
        else:
            if isinstance(query_lstm,nn.Module):
                self.query_lstm_layer = query_lstm.to(self.lm.device)
            else:
                self.query_lstm_layer = Encoder(num_inputs, self.num_hiddens, batch_first=True, bidirectional = bidirectional,  device = self.lm.device,num_layers=lstm_num_layers,normalized=normalized)
                self.query_lstm_layer.load_state_dict(torch.load(query_lstm))
        self.query_lstm_layer.to(self.lm.device)

        if corpus_lstm is None:
            self.corpus_lstm_layer = Encoder(num_inputs, self.num_hiddens, batch_first=True, bidirectional = bidirectional, device = self.lm.device,num_layers=lstm_num_layers,normalized=normalized)
        else:
            if isinstance(corpus_lstm,nn.Module):
                self.corpus_lstm_layer = corpus_lstm.to(self.lm.device)
            else:
                self.corpus_lstm_layer = Encoder(num_inputs, self.num_hiddens, batch_first=True, bidirectional = bidirectional, device = self.lm.device,num_layers=lstm_num_layers,normalized=normalized)
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
                
class Encoder(nn.Module):
    def __init__(self, num_inputs, num_hiddens, batch_first = True, bidirectional = False, device = "cuda:0",num_layers=1, normalized = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.normlized = normalized
        self.num_hiddens = num_hiddens
        self.num_directions = 2 if bidirectional else 1
        self.device = device
        # TODO: mlp & relu etc.
        self.w01 = nn.Linear(768,2048)
        self.w02 = nn.Linear(2048,1024)
        self.w03 = nn.Linear(1024,768)
        self.lstm_layter = nn.LSTM(num_inputs,self.num_hiddens,batch_first=batch_first,num_layers=num_layers)
        # self.w1 = nn.Linear(256,512)
        # self.w2 = nn.Linear(512,256)
        self.w1 = nn.Linear(256,1024)
        self.w2 = nn.Linear(1024,512)
        self.w3 = nn.Linear(512,256)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight,gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(module.bias, 0)

    def forward(self,input):
        """last_state (h_n,c_n)
        h_n: tensor of shape (D * num_layers,H_out) for unbatched input or (D * num_layers, N, H_out) containing the final hidden state for each element in the sequence."""
        batch_size = input.size(0)  # 从输入读取batch_size
        input = self.w03(self.relu(self.w02(self.relu(self.w01(input)))))
        self.state = self.begin_state(batch_size = batch_size, device=self.device)
        _,last_state = self.lstm_layter(input,self.state)
        vec = last_state[0].transpose(0,1).squeeze(1)
        output = self.w3(self.relu(self.w2(self.relu(self.w1(vec)))))
        if self.normlized:
            output = torch.nn.functional.normalize(output,dim=-1)
        if self.training:
            return output
        else:
            return output
            # return (last_state[0].transpose(0,1).repeat(1,input.shape[1],1),last_state[1])

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

