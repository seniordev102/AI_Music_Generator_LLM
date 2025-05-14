#!/bin/bash
# Train MusicBERT with masked language modeling

if [ -z ${1+x} ]; then 
    echo "data directory not set"
    exit 1
else 
    echo "data directory = ${1}"
fi

if [ -z ${2+x} ]; then
    MODEL_SIZE="base"
else
    MODEL_SIZE=${2}
fi

DATA_DIR=${1}
OUTPUT_DIR="checkpoints/mlm_${MODEL_SIZE}"

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

# Run training
python train.py \
    --task mlm \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --model-size ${MODEL_SIZE} \
    --batch-size 256 \
    --max-epochs 50 \
    --learning-rate 5e-4 \
    --num-workers 4 \
    --fp16
