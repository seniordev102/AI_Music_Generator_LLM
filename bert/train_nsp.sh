#!/bin/bash
# Train MusicBERT for next sentence prediction (NSP)

if [ -z ${1+x} ]; then echo "data directory not set" && exit 1; else echo "data directory = ${1}"; fi
if [ -z ${2+x} ]; then echo "model path not set" && exit 1; else echo "model path = ${2}"; fi

DATA_DIR=${1}
MODEL_PATH=${2}
OUTPUT_DIR="checkpoints/nsp"

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

# Run training
python train.py \
    --task nsp \
    --data-dir ${DATA_DIR} \
    --model-path ${MODEL_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --model-size base \
    --batch-size 64 \
    --max-epochs 10 \
    --learning-rate 5e-5 \
    --num-workers 4 \
    --fp16
