#!/bin/bash

# Check if exactly one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 {categorical|arousal|dominance|valence|multitask|all}"
    exit 1
fi

# Function definitions for each task
do_categorical() {
    echo "Downloading categorical model"
    wget https://lab-msp.com/MODELS/Odyssey_Baselines/weight_cat_ser.zip 
    unzip weight_cat_ser.zip
    rm weight_cat_ser.zip
}

do_arousal() {
    echo "Downloading arousal model"
    wget https://lab-msp.com/MODELS/Odyssey_Baselines/dim_aro_ser.zip 
    unzip dim_aro_ser.zip
    rm dim_aro_ser.zip
}

do_dominance() {
    echo "Downloading dominance model"
    wget https://lab-msp.com/MODELS/Odyssey_Baselines/dim_dom_ser.zip 
    unzip dim_dom_ser.zip
    rm dim_dom_ser.zip
}

do_valence() {
    echo "Downloading valence model"
    wget https://lab-msp.com/MODELS/Odyssey_Baselines/dim_val_ser.zip 
    unzip dim_val_ser.zip
    rm dim_val_ser.zip
}

do_multitask() {
    echo "Downloading multitask model"
    wget https://lab-msp.com/MODELS/Odyssey_Baselines/dim_ser.zip 
    unzip dim_ser.zip
    rm dim_ser.zip
}

# Main logic to process the input argument
for arg in "$@"
do
    case $1 in
        categorical)
            do_categorical
            ;;
        arousal)
            do_arousal
            ;;
        dominance)
            do_dominance
            ;;
        valence)
            do_valence
            ;;
        multitask)
            do_multitask
            ;;
        all)
            do_categorical
            do_arousal
            do_dominance
            do_valence
            do_multitask
            ;;
        *)
            echo "Invalid argument: $1"
            echo "Usage: $0 {categorical|arousal|dominance|valence|multitask|all}"
            exit 2
            ;;
    esac
done

exit 0
