if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
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

for pred_len in 96 192 336 720
do
for topk in 1 3 5 10
do
python -u -m torch.distributed.run --nproc_per_node=8 --master_port=2424 run.py \
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
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Prediction' \
    --train_epochs 100\
    --patience 10 \
    --topk $topk \
    --retrieval_model_path '../Retrieval/checkpoints/'$retrieval_data'/'$pred_len \
    --query_model_path '../Retrieval/checkpoints/CaRTS_Retrieval_Retrieval_weather_weather_sl96_ll48_pl'$pred_len'_dm128_nh16_el3_df256_wd0_lr0.001_normTrue_Retrieval_0_DDP' \
    --use_multi_gpu \
    --itr 1 --batch_size 1 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$topk'.log'
done
done