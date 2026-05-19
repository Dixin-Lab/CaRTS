if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/weather" ]; then
    mkdir ./logs/weather
fi

seq_len=96
model_name=Retrieval
root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=weather
retrieval_data=weather
retrieval_data_path=weather.csv
random_seed=2024

for pred_len in 96 192 336 720
do
python -u -m torch.distributed.run --nproc_per_node=6 --master_port=2424 run.py  \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id CaRTS_$model_name \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Retrieval' \
    --train_epochs 100\
    --patience 100 \
    --retrieval_data $retrieval_data \
    --retrieval_data_path $retrieval_data_path \
    --use_multi_gpu \
    --normalization True \
    --itr 1 --batch_size 128 --learning_rate 0.001 >logs/weather/$model_name'_'$model_id_name'_'$retrieval_data'_'$seq_len'_'$pred_len'.log' 
done