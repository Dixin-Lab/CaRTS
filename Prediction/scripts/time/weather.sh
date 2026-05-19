if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/time" ]; then
    mkdir ./logs/time
fi

seq_len=96
model_name=Prediction
root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=weather
retrieval_data=weather
retrieval_data_path=weather.csv
random_seed=2024
for topk in 1 3 5 10
do
python -u -m torch.distributed.run --nproc_per_node=3 --master_port=2424 run.py \
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
    --pred_len 336 \
    --enc_in 21 \
    --des 'Prediction' \
    --train_epochs 100\
    --patience 10 \
    --topk $topk \
    --retrieval_model_path '../Retrieval/checkpoints/'$retrieval_data'/336' \
    --query_model_path ../Retrieval/checkpoints/CaRTS_Retrieval_Retrieval_weather_weather_sl96_ll48_pl336_dm128_nh16_el3_df256_wd0_lr0.001_normTrue_Retrieval_0_DDP \
    --use_multi_gpu \
    --itr 1 --batch_size 1 --learning_rate 0.001 >logs/time/$model_name'_'$model_id_name'_'$seq_len'_336_'$topk'.log'
done