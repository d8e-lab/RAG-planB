current_datetime=$(date +"%m%d_%H_%M_%S")
model_path=/mnt/82_store/LLM-weights/bge-m3/
epoch=$1
lr=$3
CUDA_VISIBLE_DEVICES=$2 python myrun.py \
    --model_name_or_path $model_path \
    --train_data_file /mnt/82_store/sbc/rag_data/random_sample_3/ \
    --num_train_epochs $epoch \
    --learning_rate $lr \
    --output_dir /mnt/82_store/sbc/planB/weights/dual-bge/epoch-${epoch}/${current_datetime} \
    --do_train \
    --max_grad_norm 2.0