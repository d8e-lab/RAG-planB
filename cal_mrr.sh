current_datetime=$(date +"%m%d_%H_%M_%S")
model_path=/mnt/82_store/sbc/planB/weights/single-bert/epoch-10/0703_10_04_53/epoch-9/
CUDA_VISIBLE_DEVICES=4 python cal_mrr.py \
    --model_path $model_path \
    --test_data_file /mnt/82_store/sbc/rag_data/valid_data/ \
    --tokenizer_path /mnt/82_store/sbc/planB/weights/single-bert/epoch-10/0703_10_04_53/epoch-9/ \
    --output_dir /mnt/82_store/sbc/planB/eval_mrr/${current_datetime}