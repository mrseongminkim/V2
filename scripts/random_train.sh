#!/bin/bash

python NeuralSplitter/train.py \
    --train_path ./data/random_data/train.csv \
    --valid_path ./data/random_data/valid.csv \
    --expt_dir saved_models/random \
    --gru \
    --hidden_size 256 \
    --num_layer 2 \
    --bidirectional \
    --batch_size 512 \
    --dropout_en 0.4 \
    --dropout_de 0.4 \
    --weight_decay 0.000001 \
    --add_seed 152
