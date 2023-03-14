#!/bin/bash

cd /home/victorialin/Documents/2022-2023/causal_text/code/

treatments=(treatycommit brave evil flag threat economy treatyviolation)
conditions=(conditional_prob marginal_prob)
methods=(clf)
params=()

for method in ${methods[@]}
do
    for condition in ${conditions[@]}
    do
        if [[ $condition == "marginal_prob" ]]
        then 
            params=( --marginal-probs )
        fi
        for treatment in ${treatments[@]}
        do
            printf "Method: $method \t Prob: $condition \t Treatment: $treatment\n"
            python estimate.py --method $method --treatment $treatment "${params[@]}" --ci --lm-library sentence-transformers --lm-name all-mpnet-base-v2 
        done
        params=()
    done
done