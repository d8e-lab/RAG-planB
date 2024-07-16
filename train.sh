current_datetime=$(date +"%m%d_%H_%M_%S")
model_path=/mnt/82_store/LLM-weights/bert-base-chinese/
epoch=$1
lstm_num_layers=$3
save_path=/mnt/82_store/sbc/planB/weights/bert-lstm/bert_freezed_${lstm_num_layers}lstm_mlp_relu_normalized/${current_datetime}
# save_path=/mnt/82_store/sbc/planB/weights/original_model
# save_path=/mnt/82_store/sbc/planB/weights/bert-lstm/hack_model
data_path=/mnt/82_store/sbc/rag_data/random_sample_5/
# data_path=/mnt/82_store/sbc/rag_data/advanced_data/sample_5/
CUDA_VISIBLE_DEVICES=$2 python myrun.py \
    --model_name_or_path $model_path \
    --train_data_file $data_path \
    --num_train_epochs $epoch \
    --learning_rate 5e-6 \
    --output_dir $save_path \
    --do_train \
    --corpus_batch_size 6 \
    --freeze_lm \
    --temperature 0.002 \
    --lstm_num_layers $lstm_num_layers \
    --normalized