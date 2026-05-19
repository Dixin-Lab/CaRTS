export CUDA_VISIBLE_DEVICES=1
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/size" ]; then
    mkdir ./logs/size
fi

seq_len=96
model_name=Prediction
root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
retrieval_data=ETTh1
retrieval_data_path=ETTh1.csv
random_seed=2024
topk=1
for pred_len in 336
do
for portion in 0.05 0.2 0.5 1
do
python -u run.py \
    --random_seed $random_seed \
    --is_training 0 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --retrieval_data $retrieval_data \
    --retrieval_data_path $retrieval_data_path \
    --model_id CaRTS_$model_name \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Prediction' \
    --train_epochs 100\
    --patience 10 \
    --batch_size 1 \
    --topk $topk \
    --portion $portion \
    --retrieval_model_path '../Retrieval/checkpoints/'$retrieval_data'/'$pred_len \
    --query_model_path '../Retrieval/checkpoints/CaRTS_Retrieval_Retrieval_ETTh1_ETTh1_sl96_ll48_pl'$pred_len'_dm128_nh16_el3_df256_wd0_lr0.001_normTrue_Retrieval_0' \
    --itr 5 --learning_rate 0.0001 >logs/size/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$topk'_'$portion'.log'
done
done