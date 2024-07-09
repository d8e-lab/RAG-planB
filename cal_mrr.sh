current_datetime=$(date +"%m%d_%H_%M_%S")
for i in {0..14}; do
    model_path_query=/mnt/82_store/sbc/planB/weights/dual-bert/epoch-15/0708_06_57_29/epoch-${i}/query_bert/
    model_path_corpus=/mnt/82_store/sbc/planB/weights/dual-bert/epoch-15/0708_06_57_29/epoch-${i}/corpus_bert/
    CUDA_VISIBLE_DEVICES=4 python cal_mrr.py \
        --model_path_query $model_path_query \
        --model_path_corpus $model_path_corpus \
        --eval_data_file /mnt/82_store/sbc/rag_data/random_sample_5/ \
        --output_file=/mnt/82_store/sbc/planB/eval_mrr/${current_datetime}/epoch-15/epoch-${i}_mrr_val.txt
done

# model_path=/mnt/82_store/sbc/planB/weights/single-bert/epoch-10/0703_10_04_53/epoch-9/
# CUDA_VISIBLE_DEVICES=4 python cal_mrr.py \
#     --model_path $model_path \
#     --eval_data_file /mnt/82_store/sbc/rag_data/valid_data/ \
#     --tokenizer_path $model_path \
#     --output_dir /mnt/82_store/sbc/planB/eval_mrr/${current_datetime}

# current_datetime=$(date +"%m%d_%H_%M_%S")
