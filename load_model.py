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
# state_dict = torch.load("/mnt/82_store/sbc/planB/weights/bert-lstm/epoch-0/0710_03_45_24/torch.pt")
# print(state_dict.keys())
bert_tokenizer = BertTokenizer.from_pretrained("/mnt/82_store/LLM-weights/bert-base-chinese/",max_length=512,truncation=True,padding=True,return_tensors="pt")
model = Model(lm="/mnt/82_store/sbc/planB/weights/bert-lstm/epoch-0/0710_05_07_23/bert",pad_token_id = bert_tokenizer.pad_token_id, query_lstm="/mnt/82_store/sbc/planB/weights/bert-lstm/epoch-0/0710_05_07_23/query_lstm.pt",corpus_lstm="/mnt/82_store/sbc/planB/weights/bert-lstm/epoch-0/0710_05_07_23/corpus_lstm.pt")