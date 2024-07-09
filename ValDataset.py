from typing import Any
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import jsonlines
import torch
"""/mnt/82_store/sbc/rag_data/"""


class ValDataset(Dataset):
    def __init__(self, tokenizer, args, file_path="/mnt/82_store/sbc/rag_data/",pool=None):
        super().__init__()
        self.args=args
        self.examples = []
        import os
        from pathlib import Path
        file_list = [Path(file_path)/file for file in os.listdir(file_path) if "val" in file]
        for file in file_list:
            with jsonlines.open(file,'r') as lines:
                for line in lines:
                    query = line["query"]
                    pos = line["pos"][0]
                    neg = line["neg"]
                    query_inputs = tokenizer(query,truncation=True,padding='max_length', return_tensors="pt").input_ids
                    pos_inputs = tokenizer(pos,truncation=True,padding='max_length', return_tensors="pt").input_ids
                    neg_inputs = tokenizer(neg,truncation=True,padding='max_length', return_tensors="pt").input_ids
                    self.examples.append({
                        "query_inputs":query_inputs,
                        "pos_inputs":pos_inputs,
                        "neg_inputs":neg_inputs,
                        "atte_mask":query_inputs.ne(tokenizer.pad_token_id)
                        })

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index) -> Any:
        return self.examples[index]

def convert_nl_to_features(item):
    text, tokenizer = item
    text_tokens = tokenizer.encode(text,return_tensors="pt",padding=True)
    return text_tokens