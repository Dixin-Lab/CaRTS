export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/gpt" ]; then
    mkdir ./logs/gpt
fi

seq_len=96
model_name=GPT4TS
root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
retrieval_data=ETTm1
retrieval_data_path=ETTm1.csv
random_seed=2024
for pred_len in 192
do
for topk in 1
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
    --d_model 768 \
    --topk $topk \
    --retrieval_model_path '../Retrieval/checkpoints/'$retrieval_data'/'$pred_len \
    --query_model_path '../Retrieval/checkpoints/CaRTS_Retrieval_Retrieval_ETTm1_ETTm1_sl96_ll48_pl'$pred_len'_dm128_nh16_el3_df256_wd0_lr0.001_normTrue_Retrieval_0_DDP' \
    --itr 1 --batch_size 64 --learning_rate 0.0001 >logs/gpt/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$topk'_3layers.log'
done
done