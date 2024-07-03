current_datetime=$(date +"%m%d_%H_%M_%S")
epoch=1
CUDA_VISIBLE_DEVICES=1 python myrun.py --model_name_or_path /mnt/82_store/LLM-weights/bert-base-chinese/ --train_data_file /mnt/82_store/sbc/rag_data/random_sample_5/ --model_name_or_path /mnt/82_store/LLM-weights/bert-base-chinese/ --num_train_epochs $epoch --learning_rate 1e-5 --output_dir /mnt/82_store/sbc/planB/weights/single-bert/epoch-${epoch}/${current_datetime}