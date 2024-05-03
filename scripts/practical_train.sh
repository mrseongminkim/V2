#!/bin/bash

train="./data/practical_data/integrated/train.csv"
valid="./data/practical_data/integrated/valid.csv"

debug="./data/practical_data/integrated/test_regexlib.csv"

python NeuralSplitter/train.py \
    --train_path $train \
    --valid_path $valid \
    --expt_dir saved_models/practical \
    --gru \
    --hidden_size 256 \
    --num_layer 2 \
    --bidirectional \
    --batch_size 512 \
    --dropout_en 0.4 \
    --dropout_de 0.4 \
    --weight_decay 0.000001 \
    --add_seed 152 \
