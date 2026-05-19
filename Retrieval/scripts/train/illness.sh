if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=36
model_name=Retrieval
root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=illness
data_name=illness
retrieval_data=illness
retrieval_data_path=national_illness.csv
random_seed=2024

python -u run.py  \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id CaRTS_$model_name \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len 24 \
    --enc_in 7 \
    --des 'Retrieval' \
    --train_epochs 100\
    --patience 100 \
    --retrieval_data $retrieval_data \
    --retrieval_data_path $retrieval_data_path \
    --normalization True \
    --itr 1 --batch_size 16 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_24.log' 

python -u run.py  \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id CaRTS_$model_name \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len 36 \
    --enc_in 7 \
    --des 'Retrieval' \
    --train_epochs 100\
    --patience 100 \
    --retrieval_data $retrieval_data \
    --retrieval_data_path $retrieval_data_path \
    --normalization True \
    --itr 1 --batch_size 16 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_36.log'

python -u run.py  \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id CaRTS_$model_name \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len 48 \
    --enc_in 7 \
    --des 'Retrieval' \
    --train_epochs 100\
    --patience 100 \
    --retrieval_data $retrieval_data \
    --retrieval_data_path $retrieval_data_path \
    --normalization True \
    --itr 1 --batch_size 16 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_48.log'

python -u run.py  \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id CaRTS_$model_name \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len 60 \
    --enc_in 7 \
    --des 'Retrieval' \
    --train_epochs 100\
    --patience 100 \
    --retrieval_data $retrieval_data \
    --retrieval_data_path $retrieval_data_path \
    --normalization True \
    --itr 1 --batch_size 16 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_60.log'