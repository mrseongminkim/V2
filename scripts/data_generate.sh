#!/bin/bash

augment_size=10
random_train_data_size=100
random_valid_data_size=100

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
    python data_generator/practical_data/data_generator.py --augment $augment_size
    echo "Generation completed, Integration started."
    python data_generator/practical_data/data_integration.py
    echo "Integration completed."
    #python data_generator/practical_data/data_generator_test.py --augment 10
}

if [ $1 == "random" ]; then
    generate_random_data
else
    generate_practical_data
fi
