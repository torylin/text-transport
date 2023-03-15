#!/bin/bash

cd /work/vlin2/causal_text/code/

treatments=(treatycommit brave evil flag threat economy treatyviolation)
# conditions=(conditional_prob marginal_prob)
conditions=(conditional_prob)
# methods=(clf lm)
methods=(clm)
params=()
lm_name=/work/vlin2/causal_text/models/google/reformer-crime-and-punishment/best_model/
batch_size=8
data_dir=/work/vlin2/causal_text/data/

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
            python estimate.py --method $method --treatment $treatment "${params[@]}" --ci --lm-name $lm_name --batch-size $batch_size --data-dir $data_dir
        done
        params=()
    done
done