#!/bin/bash

# $1: [practical | random]

practical_train="./data/practical_data/integrated/train.csv"
practical_valid="./data/practical_data/integrated/valid.csv"
debug="./data/practical_data/integrated/test_snort.csv"

if [ $1 == "practical" ]; then
    train_file_path=$practical_train
    valid_file_path=$practical_valid
    expt_dir="saved_models/practical"
elif [ $1 == "random" ]; then
    train_file_path="./data/random_data/train.csv"
    valid_file_path="./data/random_data/valid.csv"
    expt_dir="saved_models/random"
elif [ $1 == "debug" ]; then
    train_file_path=$debug
    valid_file_path=$debug
    expt_dir="saved_models/practical"
fi

python RegexSplitter/train.py \
    --train_file_path $train_file_path \
    --valid_file_path $valid_file_path \
    --expt_dir $expt_dir \
    --hidden_dim 256 \
    --n_layers 2 \
    --weight_decay 0.000001 \
    --batch_size 512 \
    --rnn_type "lstm" \
    --gpu_idx 2 \
# --set_transformer \