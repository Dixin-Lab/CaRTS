export CUDA_VISIBLE_DEVICES=1
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=36
model_name=Prediction
root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=illness
data_name=illness
retrieval_data=illness
retrieval_data_path=national_illness.csv
random_seed=2024
for pred_len in 24 36 48 60
do
for topk in 1 3 5 10
do
python -u run.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --retrieval_data $retrieval_data \
    --retrieval_data_path $retrieval_data_path \
    --model_id CaRTS_$model_name \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Prediction' \
    --train_epochs 100\
    --patience 10 \
    --topk $topk \
    --retrieval_model_path '../Retrieval/checkpoints/'$retrieval_data'/'$pred_len \
    --query_model_path '../Retrieval/checkpoints/CaRTS_Retrieval_Retrieval_illness_illness_sl36_ll18_pl'$pred_len'_dm128_nh16_el3_df256_wd0_lr0.001_normTrue_Retrieval_0' \
    --itr 1 --batch_size 16 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$topk'_lr0.001.log'
done
done