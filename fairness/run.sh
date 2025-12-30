#!/bin/bash

NOR=$1
UNF=$2
DATASETS=("census" "bank" "meps15" "compas")
for dataset in "${DATASETS[@]}"; do
    case $dataset in
        "census")
            attrs=("age" "race" "sex")
            ;;
        "bank")
            attrs=("age")
            ;;
        "meps15")
            attrs=("age" "race" "sex")
            ;;
        "compas")
            attrs=("age" "race")
            ;;
    esac

    for sa in "${attrs[@]}"; do
    
        echo "----------------------------------------------------------"
        echo "Run: Dataset=$dataset, Net=$net, SA=$sa
        echo "Run: python main.py --net $net --dataset $dataset --SA $sa"
        echo "----------------------------------------------------------"
        
        python main.py --dataset "$dataset" --SA "$sa" --log --N_normal "$NOR" --N_unfair "$UNF"
        
    done
done

echo "所有实验任务已完成！"