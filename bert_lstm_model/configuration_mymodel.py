from transformers import PretrainedConfig
from typing import List


class MyModelConfig(PretrainedConfig):


    def __init__(
        self,
        lstm_path="/mnt/82_store/sbc/planB/weights/bert-lstm/epoch-0/0710_05_07_23/query_lstm.pt",
        bert_path="/mnt/82_store/sbc/planB/weights/bert-lstm/epoch-0/0710_05_07_23/bert",
        bert_base="/mnt/82_store/LLM-weights/bert-base-chinese",
        embedding_type="query",
        **kwargs,
    ):
        self.lstm_path = lstm_path
        self.bert_path = bert_path
        self.bert_base = bert_base
        self.embedding_type = embedding_type
        super().__init__(**kwargs)


