import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
import torch 
from model import Model
from transformers import BertModel, BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from info_nce import InfoNCE, info_nce
from TextDataset import TextDataset
from pathlib import Path
from torch import nn

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_sim(model:Model,query_input_ids,corpus_inputs,args):
    model.set_query_mode()
    query_outputs=model(input_ids=query_input_ids)
    model.set_corpus_mode()
    corpus_outpus=model(input_ids=corpus_inputs)
    scores = torch.matmul(corpus_outpus,query_outputs.transpose(-2,-1)).view(query_outputs.shape[0],-1)/ args.temperature # temperature
    return scores

def train(args):
    logger = logging.getLogger(__name__)
    device = "cuda:0"
    bert = BertModel.from_pretrained(args.model_name_or_path).to(device)
    bert_tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,max_length=512,truncation=True,padding=True,return_tensors="pt")
    model = Model(lm=bert, pad_token_id = bert_tokenizer.pad_token_id, corpus_batch_size = args.corpus_batch_size, lm_freezed = args.freeze_lm,lstm_num_layers=args.lstm_num_layers,normalized=args.normalized,bidirectional=args.bidirectional)
    dataset = TextDataset(bert_tokenizer,args,file_path=args.train_data_file)
    train_sampler = RandomSampler(dataset)
    model.zero_grad()

    train_dataloader = DataLoader(dataset,sampler=train_sampler,batch_size=args.train_batch_size,num_workers=4)
    model.train()
    tr_num,tr_loss,best_mrr=0,0,0 
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            query_inputs = batch["query_inputs"].to(args.device)[0]
            pos_inputs = batch["pos_inputs"].to(args.device)[0]
            neg_inputs = batch["neg_inputs"].to(args.device)[0]
            corpus_inputs = torch.cat((pos_inputs,neg_inputs),dim=0)
            scores = calculate_sim(model,query_inputs,corpus_inputs,args)
            # loss_fct = InfoNCE(negative_mode='unpaired')
            # loss = loss_fct(query_outputs,pos_outputs,neg_outputs)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores,torch.zeros(scores.shape[0],device=scores.device, dtype=torch.long))
            #report loss
            tr_loss += loss.item()
            tr_num+=1
            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss=0
                tr_num=0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
        save_path = Path(args.output_dir)/f'epoch-{idx}'
        # save_path = Path(args.output_dir)
        model.save(save_path,bert_tokenizer)
        # bert_tokenizer.save_pretrained(save_path)
        # bert_tokenizer.save_vocabulary(save_path)
    if args.num_train_epochs==0:
        save_path = Path(args.output_dir)
        model.save(save_path,bert_tokenizer)
    ## MRR eval()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--eval_data_file", default=None, type=str,
    #                     help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    # parser.add_argument("--test_data_file", default=None, type=str,
    #                     help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--corpus_batch_size", default=6, type=int,
                        help="N negative samples + 1 positive sample")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--freeze_lm", action='store_true',
                        help="Whether to freeze language model when training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--temperature', type=float, default=0.02,
                        help="Set temperature for sim_score to scale up loss.")
    parser.add_argument('--lstm_num_layers', type=int, default=1,
                        help="lstm_num_layers")
    parser.add_argument("--normalized", action='store_true')
    parser.add_argument("--bidirectional", action='store_true')
    
    args = parser.parse_args()
    
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)


    train(args)

if __name__=="__main__":
    main()