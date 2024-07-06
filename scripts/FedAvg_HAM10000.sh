#!/bin/bash
output_dir="outputs"

python train_fedmedicl.py \
--datasets HAM10000 \
--algorithm fedavg \
--num_clients 10 \
--num_tasks 4 \
--num_rounds 150 \
--num_iters 5 \
--training_log_frequency 100 \
--average_all_layers \
--batch_size 10 \
--num_evaluations 3 \
--num_workers 3 \
--task_split_type group_probability \
--is_imbalanced \
--imbalance_ratios "{\"balanced\": 0.2, \"spare\": 0.2}" \
--output_dir $output_dir 