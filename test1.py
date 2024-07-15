from sentence_transformers import SentenceTransformer
from transformers import AutoModel,AutoTokenizer
from bert_lstm_model.modeling_mymodel import MyModel
from bert_lstm_model.configuration_mymodel import MyModelConfig
import torch
from transformers import BertTokenizer
# berttokenizer = AutoTokenizer.from_pretrained("/mnt/82_store/sbc/planB/weights/bert-lstm/bert_freezed_mask_pad/0711_14_11_54/epoch-0/bert/")
# model = AutoModel.from_pretrained("/mnt/82_store/sbc/planB/weights/bert-lstm/bert_freezed_mask_pad/0711_14_11_54/epoch-0/bert/",trust_remote_code=True)
model1 = SentenceTransformer("/mnt/82_store/sbc/planB/weights/bert-lstm/bert_freezed_mask_pad/0711_14_11_54/epoch-0/bert/",trust_remote_code=True)


# corpus_model_config = MyModelConfig.from_json_file("corpus_bert_lstm_model/config.json")
# query_model = MyModel(corpus_model_config)
query_model = AutoModel.from_pretrained("/mnt/82_store/sbc/planB/weights/bert-lstm/bert_freezed_mask_pad/0711_14_11_54/epoch-0/bert/").to("cuda")
a = ["www"]

berttokenizer = BertTokenizer.from_pretrained("/mnt/82_store/LLM-weights/bert-base-chinese/",return_tensors="pt")
a = berttokenizer(a)
print(a)
b = query_model(torch.tensor(a.input_ids).to("cuda"))
print(b)
# a = ["www","A","B","C","D","D"]
# model2 = SentenceTransformer("/home/xmu/topk_add/RAG_planB/test_corpus",trust_remote_code=True)
# c = model1.encode(a,normalize_embeddings=True)
# d = model2.encode(a)
# print(d.shape)

