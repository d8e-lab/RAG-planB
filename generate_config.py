import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Generate JSON configuration for BERT and LSTM paths.")
    parser.add_argument('--bert_path', type=str, required=True, help='Path to the BERT model.')
    parser.add_argument('--embedding_type', type=str, required=True, help='Type of embeddings to use.')
    parser.add_argument('--lstm_path', type=str, required=True, help='Path to the LSTM model.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the JSON configuration file.')

    args = parser.parse_args()

    config = {
        "bert_path": args.bert_path,
        "lstm_path": args.lstm_path,
        "embedding_type": args.embedding_type,
        "transformers_version": "4.42.3"
    }

    with open(args.save_path, 'w') as json_file:
        json.dump(config, json_file, indent=4)

if __name__ == "__main__":
    main()
