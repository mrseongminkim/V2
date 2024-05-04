#!/bin/bash

augment_size=10
random_train_data_size=400000
random_valid_data_size=1000

generate_random_data() {
    echo "Generation started."
    for alphabet_size in 2 4 6 8 10
    do
        python data_generator/random_data/data_generator.py --alphabet_size $alphabet_size --is_train --number $random_train_data_size &
        python data_generator/random_data/data_generator.py --alphabet_size $alphabet_size --number $random_valid_data_size &
        echo "Dataset for alphabet_size '$alphabet_size' is being generated in the background!"
    done
    wait
    echo "Generation completed, Integration started."
    python data_generator/random_data/data_integration.py
    echo "Integration completed."
}

generate_practical_data() {
    echo "Generation started."
    for data_name in snort-clean regexlib-clean
    do
        python data_generator/practical_data/data_generator.py $data_name --augment $augment_size &
    done
    for i in {01..54}
    do
        data_name=$(printf "practical_regexes_%s" "$i")
        python data_generator/practical_data/data_generator.py $data_name --augment $augment_size &
    done
    wait
    echo "Generation completed, Integration started."
    python data_generator/practical_data/data_integration.py
    echo "Integration completed."
}

if [ $1 == "random" ]; then
    generate_random_data
else
    generate_practical_data
fi
