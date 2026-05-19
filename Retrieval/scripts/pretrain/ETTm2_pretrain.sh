if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/pretrain" ]; then
    mkdir ./logs/LongForecasting/pretrain
fi

seq_len=96
model_name=Retrieval
root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
random_seed=2024

for pred_len in 96 192 336 720
do
    python -u run.py \
      --random_seed $random_seed \
      --is_training 0 \
      --is_pretraining 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id CaRTS_$model_name \
      --model $model_name \
      --data $data_name \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Pretrain' \
      --train_epochs 100\
      --patience 100\
      --itr 1 --batch_size 128 --learning_rate 0.001  >logs/LongForecasting/pretrain/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'.log' 
done