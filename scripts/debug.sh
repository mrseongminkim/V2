#!/bin/bash

# $1: [snort | lib | practical | 2 | 4 | 6 | 8 | 10]
# $2: [ar | bf]
time_limit=3

if [ $1 == "snort" ]; then
    data_path="./data/practical_data/integrated/test_snort.csv"
    log_path="./log_data/snort/"
    checkpoint_pos="./saved_models/practical/gru__256__2__2"
    data_type="practical"
    num=100
elif [ $1 == "lib" ]; then
    data_path="./data/practical_data/integrated/test_regexlib.csv"
    log_path="./log_data/regexlib/"
    checkpoint_pos="./saved_models/practical/gru__256__2__2"
    data_type="practical"
    num=500
elif [ $1 == "practical" ]; then
    data_path="./data/practical_data/integrated/test_practicalregex.csv"
    log_path="./log_data/practicalregex/"
    checkpoint_pos="./saved_models/practical/gru__256__2__2"
    data_type="practical"
    num=3000
fi
case $1 in "2"|"4"|"6"|"8"|"10")
    data_path="./data/random_data/size_${1}/test.csv"
    log_path="./log_data/random/${1}/"
    checkpoint_pos="./saved_models/random/gru__256__2__2"
    data_type="random"
    alphabet_size=$1
esac

if [ $2 == "ar" ]; then
    log_path+="alpharegex"
    sub_model="alpharegex"
elif [ $2 == "bf" ]; then
    log_path+="blue_fringe"
    sub_model="blue_fringe"
else
    sub_model="error"
fi

python debug.py --num $num --path $log_path --time_limit $time_limit





# Figure 2
# python debug.py --path log_data/random/2/sequential/alpharegex --time_limit 3 --num 1000 
# python debug.py --path log_data/random/4/sequential/alpharegex --time_limit 3 --num 1000 
# python debug.py --path log_data/random/6/sequential/alpharegex --time_limit 3 --num 1000 
# python debug.py --path log_data/random/8/sequential/alpharegex --time_limit 3 --num 1000 
# python debug.py --path log_data/random/10/sequential/alpharegex --time_limit 3 --num 1000 



# python debug_total.py --path log_data/regexlib/sequential/second/alpharegex --time_limit 3 --num 3000
# python debug_total.py --path log_data/snort/sequential/second/alpharegex --time_limit 3 --num 3000
# python debug_total.py --path log_data/practicalregex/sequential/second/alpharegex --time_limit 3 --num 3000

# python debug_total.py --path log_data/regexlib/sequential/third/alpharegex --time_limit 3 --num 3000
# python debug_total.py --path log_data/snort/sequential/third/alpharegex --time_limit 3 --num 3000
# python debug_total.py --path log_data/practicalregex/sequential/third/alpharegex --time_limit 3 --num 3000



# Table 3
# python debug.py --path log_data/random/10/sequential/alpharegex --time_limit 1 --num 1000 --exclude_GT
# python debug.py --path log_data/regexlib/sequential/alpharegex --time_limit 1 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/sequential/alpharegex --time_limit 1 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/sequential/alpharegex --time_limit 1 --num 3000 --exclude_GT

# python debug.py --path log_data/random/10/sequential/alpharegex --time_limit 3 --num 1000 --exclude_GT
# python debug.py --path log_data/regexlib/sequential/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/sequential/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/sequential/alpharegex --time_limit 3 --num 3000 --exclude_GT

# python debug.py --path log_data/random/10/sequential/alpharegex --time_limit 5 --num 1000 --exclude_GT
# python debug.py --path log_data/regexlib/sequential/alpharegex --time_limit 5 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/sequential/alpharegex --time_limit 5 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/sequential/alpharegex --time_limit 5 --num 3000 --exclude_GT

# python debug.py --path log_data/random/10/sequential/alpharegex --time_limit 10 --num 1000 --exclude_GT
# python debug.py --path log_data/regexlib/sequential/alpharegex --time_limit 10 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/sequential/alpharegex --time_limit 10 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/sequential/alpharegex --time_limit 10 --num 3000 --exclude_GT



# Table 4
# python debug.py --path log_data/regexlib/sequential/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/sequential/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/sequential/alpharegex --time_limit 3 --num 3000 --exclude_GT

# python debug.py --path log_data/regexlib/prefix/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/prefix/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/prefix/alpharegex --time_limit 3 --num 3000 --exclude_GT

# python debug.py --path log_data/regexlib/parallel/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/snort/parallel/alpharegex --time_limit 3 --num 3000 --exclude_GT
# python debug.py --path log_data/practicalregex/parallel/alpharegex --time_limit 3 --num 3000 --exclude_GT



# # A.3
# python debug.py --path log_data/regex_perturb/alpharegex --time_limit 3


# # A.4
# python debug.py --path log_data/regexlib/sequential/regex_generator --time_limit 15 --num 3000
# python debug.py --path log_data/snort/sequential/regex_generator --time_limit 15 --num 3000
# python debug.py --path log_data/practicalregex/sequential/regex_generator --time_limit 15 --num 3000


# # A.5
# python debug.py --path log_data/KB13_full/alpharegex --time_limit 3 
# python debug.py --path log_data/NL-RX-Turk_full/alpharegex --time_limit 3

