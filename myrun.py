import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import Model
from transformers import BertModel, BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from info_nce import InfoNCE, info_nce
from TextDataset import TextDataset
from pathlib import Path

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
def generate(model,query_input_ids,pos_inputs,neg_inputs,attn_mask=None,pad_token_id=None):
    # attn_mask = query_input_ids.ne(pad_token_id) if attn_mask is None else attn_mask
    # output = model(input_ids=query_input_ids,attention_mask=attn_mask)
    query_outputs=model(query_input_ids=query_input_ids)
    pos_outputs=model(corpus_intput_ids=pos_inputs)
    neg_outputs=model(corpus_intput_ids=neg_inputs)
    return query_outputs,pos_outputs,neg_outputs

def train(args):
    logger = logging.getLogger(__name__)
    device = "cuda:0"
    query_bert = BertModel.from_pretrained("/mnt/82_store/LLM-weights/bert-base-chinese/").to(device)
    corpus_bert = BertModel.from_pretrained("/mnt/82_store/LLM-weights/bert-base-chinese/").to(device)
    bert_tokenizer = BertTokenizer.from_pretrained("/mnt/82_store/LLM-weights/bert-base-chinese/",max_length=512,truncation=True,padding=True,return_tensors="pt")
    model = Model(query_encoder=query_bert,corpus_encoder=corpus_bert,pad_token_id = bert_tokenizer.pad_token_id)
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

            # query_outputs = model(query_input_ids=query_inputs[0])
            # pos_outputs = model(query_input_ids=pos_inputs[0])
            # neg_outputs = model(query_input_ids=neg_inputs[0]) 
            query_outputs,pos_outputs,neg_outputs = generate(model,query_inputs,pos_inputs,neg_inputs)
            loss_fct = InfoNCE(negative_mode='unpaired')
            loss = loss_fct(query_outputs,pos_outputs,neg_outputs)
            
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
        model.save(save_path,bert_tokenizer)
        # bert_tokenizer.save_pretrained(save_path)
        # bert_tokenizer.save_vocabulary(save_path)

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
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
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