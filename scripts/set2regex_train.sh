#!/bin/bash

# $1: [practical | random | debug]

practical_train="./data/practical_data/integrated/train.csv"
practical_valid="./data/practical_data/integrated/valid.csv"
debug="./data/practical_data/integrated/test_snort.csv"

if [ $1 == "practical" ]; then
    train_file_path=$practical_train
    valid_file_path=$practical_valid
    expt_dir="./saved_models/set2regex/practical"
elif [ $1 == "random" ]; then
    train_file_path="./data/random_data/train.csv"
    valid_file_path="./data/random_data/valid.csv"
    expt_dir="./saved_models/set2regex/random"
elif [ $1 == "debug" ]; then
    train_file_path=$debug
    valid_file_path=$debug
    expt_dir="./saved_models/set2regex/debug"
fi

python set2regex/train.py \
    --train_path $train_file_path \
    --valid_path $valid_file_path \
    --expt_dir $expt_dir \
    --hidden_size 256 \
    --num_layer 2 \
    --batch_size 512 \
    --gpu_idx 0 \
    --rnn_cell lstm \
    #--attn_mode \
    #--set_transformer \

# if attn_mode true
# it attend on negative examples too
