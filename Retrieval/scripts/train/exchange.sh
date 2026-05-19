if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/exchange" ]; then
    mkdir ./logs/exchange
fi

seq_len=96
model_name=Retrieval
root_path_name=./dataset/
data_path_name=exchange_rate.csv
model_id_name=exchange
data_name=exchange
retrieval_data=exchange
retrieval_data_path=exchange_rate.csv
random_seed=2024

for pred_len in 96 192 336 720
do
python -u run.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id CaRTS_$model_name \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 8 \
    --des 'Retrieval' \
    --train_epochs 100\
    --patience 100 \
    --retrieval_data $retrieval_data \
    --retrieval_data_path $retrieval_data_path \
    --normalization True \
    --itr 1 --batch_size 32 --learning_rate 0.001 >logs/exchange/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'.log' 
done