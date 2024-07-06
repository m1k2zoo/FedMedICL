#!/bin/bash
output_dir="outputs"

python train_fedmedicl.py \
--datasets "HAM10000,fitzpatrick17k,OL3I" \
--algorithm fedavg \
--num_clients 60 \
--num_tasks 2 \
--num_rounds 200 \
--num_iters 5 \
--training_log_frequency 50 \
--batch_size 10 \
--num_evaluations 2 \
--num_workers 3 \
--task_split_type 'group_probability' \
--is_imbalanced \
--imbalance_ratios "{\"balanced\": 0.2, \"spare\": 0.2}" \
--output_dir $output_dir 