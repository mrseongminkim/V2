#!/bin/bash

# $1: [snort | lib | practical | 2 | 4 | 6 | 8 | 10]
# $2: [ar | bf]
alphabet_size=5
time_limit=3

if [ $1 == "snort" ]; then
    data_path="./data/practical_data/integrated/test_snort.csv"
    log_path="./log_data/snort/"
    checkpoint_pos="./saved_models/practical/gru__256__2__2"
    data_type="practical"
elif [ $1 == "lib" ]; then
    data_path="./data/practical_data/integrated/test_regexlib.csv"
    log_path="./log_data/regexlib/"
    checkpoint_pos="./saved_models/practical/gru__256__2__2"
    data_type="practical"
elif [ $1 == "practical" ]; then
    data_path="./data/practical_data/integrated/test_practicalregex.csv"
    log_path="./log_data/practicalregex/"
    checkpoint_pos="./saved_models/practical/gru__256__2__2"
    data_type="practical"
fi
case $1 in "2"|"4"|"6"|"8"|"10")
    data_path="./data/random_data/size_${1}/test.csv"
    log_path="./log_data/random/${1}/"
    checkpoint_pos="./saved_models/random/gru__256__2__2"
    data_type="random"
    alphabet_size=$1
esac

if [ $2 == "ar" ]; then
    sub_model="alpharegex"
elif [ $2 == "bf" ]; then
    sub_model="blue_fringe"
elif [ $2 == "rg" ]; then
    sub_model="regex_generator"
else
    sub_model="error"
fi

python synthesis.py \
    --data_path $data_path \
    --log_path $log_path \
    --checkpoint_pos $checkpoint_pos \
    --data_type $data_type \
    --sub_model $sub_model \
    --time_limit $time_limit \
    --alphabet_size $alphabet_size
