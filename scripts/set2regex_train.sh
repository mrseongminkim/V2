#!/bin/bash

# $1: [practical | random]

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
fi

python RegexSplitter/train.py \
    --train_file_path $train_file_path \
    --valid_file_path $valid_file_path \
    --expt_dir $expt_dir \
    --regex_max_length 100 \
    --batch_size 128 \
    --teacher_forcing_ratio 0.5 \
    --n_layers 2 \
    --hidden_dim 512 \
    --n_epochs 999999999999 \
    --clip 1 \
    --rnn_type "gru"
