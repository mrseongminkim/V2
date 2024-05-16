#!/bin/bash

# $1: [practical | random | debug]

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

python NeuralSplitter/train.py \
    --train_path $train_file_path \
    --valid_path $valid_file_path \
    --expt_dir $expt_dir \
    --hidden_size 256 \
    --num_layer 2 \
    --batch_size 256 \
    --gpu_idx 3 \
    --rnn_cell gru \
    --set_transformer \
