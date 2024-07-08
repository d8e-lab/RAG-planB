from typing import Any
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import jsonlines
import torch
"""/mnt/82_store/sbc/rag_data/"""


class TextDataset(Dataset):
    def __init__(self, tokenizer=None, args=None, file_path="/mnt/82_store/sbc/rag_data/",pool=None):
        super().__init__()
        self.args=args
        self.examples = []
        import os
        from pathlib import Path
        file_list = [Path(file_path)/file for file in os.listdir(file_path) if "train" in file]
        for file in file_list:
            with jsonlines.open(file,'r') as lines:
                for line in lines:
                    query = line["query"]
                    pos = line["pos"][0]
                    neg = line["neg"]
                    if tokenizer is not None:
                        query_inputs = tokenizer(query,truncation=True,padding='max_length', return_tensors="pt").input_ids
                        query_inputs = query
                        pos_inputs = tokenizer(pos,truncation=True,padding='max_length', return_tensors="pt").input_ids
                        neg_inputs = tokenizer(neg,truncation=True,padding='max_length', return_tensors="pt").input_ids
                    else :
                        query_inputs = query
                        pos_inputs = pos
                        neg_inputs = neg
                    self.examples.append({
                        "query_inputs":query_inputs,
                        "pos_inputs":pos_inputs,
                        "neg_inputs":neg_inputs,
                        })

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index) -> Any:
        return self.examples[index]

