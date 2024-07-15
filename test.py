from bert_lstm_model.configuration_mymodel import MyModelConfig
from bert_lstm_model.modeling_mymodel import MyModel

MyModelConfig.register_for_auto_class()
MyModel.register_for_auto_class()

query_model_config = MyModelConfig.from_json_file("corpus_bert_lstm_model/config.json")
query_model = MyModel(query_model_config)

query_model.save_pretrained("test_corpus")


query_model_config = MyModelConfig.from_json_file("query_bert_lstm_model/config.json")
query_model = MyModel(query_model_config)

query_model.save_pretrained("test_query")
