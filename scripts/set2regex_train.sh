#!/bin/bash

# $1: [practical | random]

practical_train="./data/practical_data/integrated/train.csv"
practical_valid="./data/practical_data/integrated/valid.csv"
debug="./data/practical_data/integrated/test_snort.csv"

if [ $1 == "practical" ]; then
    train_path=$practical_train
    valid_path=$practical_valid
    expt_dir="./saved_models/set_regex/practical"
elif [ $1 == "random" ]; then
    train_path="./data/random_data/train.csv"
    valid_path="./data/random_data/valid.csv"
    expt_dir="./saved_models/set_regex/random"
fi

python set2regex/train.py \
    --train_path $train_path \
    --valid_path $valid_path \
    --expt_dir $expt_dir \
    --gru \
    --hidden_size 256 \
    --num_layer 2 \
    --bidirectional \
    --batch_size 512 \
    --dropout_en 0.4 \
    --dropout_de 0.4 \
    --weight_decay 0.000001 \
    --add_seed 152
