current_datetime=$(date +"%m%d_%H_%M_%S")
model_path=/mnt/82_store/LLM-weights/bert-base-chinese/
epoch=$1
CUDA_VISIBLE_DEVICES=$2 python myrun.py \
    --model_name_or_path $model_path \
    --train_data_file /mnt/82_store/sbc/rag_data/random_sample_5/ \
    --num_train_epochs $epoch \
    --learning_rate 5e-6 \
    --output_dir /mnt/82_store/sbc/planB/weights/dual-bert/epoch-${epoch}/${current_datetime} \
    --do_train