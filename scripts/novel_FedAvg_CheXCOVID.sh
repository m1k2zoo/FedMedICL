#!/bin/bash
output_dir="outputs"

python train_fedmedicl.py \
--datasets COVID \
--algorithm fedavg \
--num_clients 5 \
--num_tasks 4 \
--num_rounds 150 \
--num_iters 5 \
--training_log_frequency 50 \
--average_all_layers \
--batch_size 10 \
--num_evaluations 1 \
--num_workers 3 \
--is_novel_disease \
--output_dir $output_dir