if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ETTm2" ]; then
    mkdir ./logs/ETTm2
fi

seq_len=96
model_name=Retrieval
root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
retrieval_data=ETTm2
retrieval_data_path=ETTm2.csv
random_seed=2024

for pred_len in 96 192 336 720
do
python -u -m torch.distributed.run --nproc_per_node=6 --master_port=2424 run.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id CaRTS_$model_name \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Retrieval' \
    --train_epochs 100\
    --patience 100 \
    --retrieval_data $retrieval_data \
    --retrieval_data_path $retrieval_data_path \
    --normalization True \
    --use_multi_gpu \
    --itr 1 --batch_size 128 --learning_rate 0.001 >logs/ETTm2/$model_name'_'$model_id_name'_'$retrieval_data'_'$seq_len'_'$pred_len'.log' 
done

