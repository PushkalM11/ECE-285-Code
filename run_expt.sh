#!/bin/bash
# Usage: ./run_expt.sh <gpu_id>
python3 EmerDiff.py \
    --image_id 21132 \
    --gpu_id "$1" \
    --text_prompt "" \
    --num_mask 25 \
    --lambda_1 -10 \
    --lambda_2 10 \
    --inference_steps 50 \
    --modulation_time_step 281 \
    --save_dir "/home/pushkalm11/Courses/ece285/Project/results/"

# Working Images:
# 1. 00068
# 2. 21132
