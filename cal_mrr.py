import argparse
import logging
import statistics
import os
import pickle
import random
import torch
import json
import numpy as np
import torch.nn.functional as F
from model import Model
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from ValDataset import ValDataset
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

def mean_reciprocal_rank(results):
    for i, r in enumerate(results):
        if r == 0:
            return 1.0 / (i + 1)
    return 0.0

##MRR
def eval(args):
    logger = logging.getLogger(__name__)
    device = "cuda:0"

    query_bert = BertModel.from_pretrained(args.model_path_query).to(device)
    corpus_bert = BertModel.from_pretrained(args.model_path_corpus).to(device)
    bert_tokenizer = BertTokenizer.from_pretrained("/mnt/82_store/LLM-weights/bert-base-chinese/",max_length=512,truncation=True,padding=True,return_tensors="pt")

    model = Model(query_encoder=query_bert,corpus_encoder=corpus_bert,pad_token_id = bert_tokenizer.pad_token_id)
    dataset = ValDataset(bert_tokenizer,args,file_path=args.eval_data_file)
    eval_dataloader = DataLoader(dataset,batch_size=args.eval_batch_size,num_workers=4)

    mrr = []

    for batch in eval_dataloader:
        #get inputs
        query_inputs = batch["query_inputs"].to(args.device)[0]
        pos_inputs = batch["pos_inputs"].to(args.device)[0]
        neg_inputs = batch["neg_inputs"].to(args.device)[0]

        query_outputs,pos_outputs,neg_outputs = generate(model,query_inputs,pos_inputs,neg_inputs)
        pos_and_neg = torch.cat((pos_outputs, neg_outputs), dim=0)
        cosine_sim = F.cosine_similarity(query_outputs, pos_and_neg, dim=1).tolist()

        # 对列表进行排序并保留原始索引
        results = [index for index, value in sorted(enumerate(cosine_sim), key=lambda x: x[1], reverse=True)]
        mrr.append(mean_reciprocal_rank(results))
    
        #loss_fct = InfoNCE(negative_mode='unpaired')
        #loss = loss_fct(query_outputs,pos_outputs,neg_outputs)
        #save_path = Path(args.output_dir)/f'epoch-{idx}'

    # 计算均值和方差
    mean_value = statistics.mean(mrr)
    variance_value = statistics.variance(mrr)
    mrr = [mean_value, variance_value] + mrr

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args.output_file, 'w') as file:
        for item in mrr:
            file.write(f"{item}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--model_path_query", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--model_path_corpus", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_path", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size for evaluation.")
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


    eval(args)

if __name__=="__main__":
    main()