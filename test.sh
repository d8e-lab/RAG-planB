current_datetime=$(date +"%m%d_%H_%M_%S")
model_path=/mnt/82_store/sbc/planB/weights/bert-lstm/bert_freezed_3lstm_mlp_relu_normalized/0716_05_36_30/epoch-7
epoch=$1
lstm_num_layers=$3
# /mnt/82_store/sbc/rag_data/random_sample_5/
CUDA_VISIBLE_DEVICES=$2 python local_test.py \
    --model_name_or_path $model_path \
    --train_data_file /mnt/82_store/sbc/rag_data/test/ \
    --num_train_epochs $epoch \
    --learning_rate 1e-6 \
    --output_dir /mnt/82_store/sbc/planB/weights/bert-lstm/bert_nofreezed_mask_pad/${current_datetime} \
    --do_test \
    --corpus_batch_size 6 \
    --temperature 0.02 \
    --normalized \
    --lstm_num_layers $lstm_num_layers
    # --freeze_lm \