#!/bin/bash

# $1: [snort | lib | practical | 2 | 4 | 6 | 8 | 10]
# $2: [ar | bf | rg | sr]
# $3: [basic | parallel | prefix]
alphabet_size=5

if [ $1 == "snort" ]; then
    data_path="./data/practical_data/integrated/test_snort.csv"
    log_path="./log_data/snort/"
    checkpoint_pos="./saved_models/practical/lstm_256_2_False/best_accuracy/checkpoints/2024_05_24_16_47_44"
    data_type="practical"
elif [ $1 == "lib" ]; then
    data_path="./data/practical_data/integrated/test_regexlib.csv"
    log_path="./log_data/regexlib/"
    checkpoint_pos="./saved_models/practical/lstm_256_2_False/best_accuracy/checkpoints/2024_05_24_16_47_44"
    data_type="practical"
elif [ $1 == "practical" ]; then
    data_path="./data/practical_data/integrated/test_practicalregex.csv"
    log_path="./log_data/practicalregex/"
    checkpoint_pos="./saved_models/practical/lstm_256_2_False/best_accuracy/checkpoints/2024_05_24_16_47_44"
    data_type="practical"
fi
case $1 in "2"|"4"|"6"|"8"|"10")
    data_path="./data/random_data/size_${1}/test.csv"
    log_path="./log_data/random/${1}/"
    checkpoint_pos="./saved_models/random/gru_256_2_True/best_accuracy/checkpoints/2024_05_24_17_33_35"
    data_type="random"
    alphabet_size=$1
esac

if [ $2 == "ar" ]; then
    sub_model="alpharegex"
elif [ $2 == "bf" ]; then
    sub_model="blue_fringe"
elif [ $2 == "rg" ]; then
    sub_model="regex_generator"
elif [ $2 == "sr" ]; then
    sub_model="set2regex"
else
    sub_model="error"
fi

if [ $3 == "basic" ]; then
    synthesis_strategy="sequential_basic"
elif [ $3 == "parallel" ]; then
    synthesis_strategy="parallel"
elif [ $3 == "prefix" ]; then
    synthesis_strategy="sequential_prefix"
else
    synthesis_strategy="error"
fi

python synthesis.py \
    --data_path $data_path \
    --log_path $log_path \
    --checkpoint_pos $checkpoint_pos \
    --data_type $data_type \
    --sub_model $sub_model \
    --alphabet_size $alphabet_size \
    --synthesis_strategy $synthesis_strategy \
