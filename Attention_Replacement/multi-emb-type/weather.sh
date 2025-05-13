if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=PatchTST

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021
pred_len=96
train_epochs=10
for EMBTYPE in linear conv mlp residualblock
# for EMBTYPE in conv
do
    export EMBTYPE=$EMBTYPE
    echo "Current EMBTYPE: $EMBTYPE"
    # 两种ATTENTYPE，RAW和MEAN
    for ATTENTYPE in RAW MEAN
    do
        # 设置ATTENTYPE为环境变量，变量名为ATTENTYPE
        export ATTENTYPE=$ATTENTYPE
        echo "Current ATTENTYPE: $ATTENTYPE"
        python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name_$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 21 \
        --e_layers 3 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --patch_len 16\
        --stride 16\
        --des 'Exp' \
        --train_epochs $train_epochs\
        --patience 20\
        --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$EMBTYPE'_'$ATTENTYPE.log
    done
done