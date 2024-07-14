import argparse
from bert_lstm_model.configuration_mymodel import MyModelConfig
from bert_lstm_model.modeling_mymodel import MyModel

def main():
    parser = argparse.ArgumentParser(description="Process paths for model configuration and saving.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the JSON configuration file.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the pretrained model.')

    args = parser.parse_args()

    MyModelConfig.register_for_auto_class()
    MyModel.register_for_auto_class()

    query_model_config = MyModelConfig.from_json_file(args.config_path)
    query_model = MyModel(query_model_config)

    query_model.save_pretrained(args.save_path)

if __name__ == "__main__":
    main()
