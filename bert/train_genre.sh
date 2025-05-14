#!/bin/bash
# Train MusicBERT for genre classification

if [ -z ${1+x} ]; then echo "data directory not set" && exit 1; else echo "data directory = ${1}"; fi
if [ -z ${2+x} ]; then echo "model path not set" && exit 1; else echo "model path = ${2}"; fi
if [ -z ${3+x} ]; then MODEL_SIZE="base"; else MODEL_SIZE=${3}; fi

DATA_DIR=${1}
MODEL_PATH=${2}
OUTPUT_DIR="checkpoints/genre_${MODEL_SIZE}"

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

# Run training
python train.py \
    --task genre \
    --data-dir ${DATA_DIR} \
    --model-path ${MODEL_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --model-size ${MODEL_SIZE} \
    --batch-size 32 \
    --max-epochs 20 \
    --learning-rate 5e-5 \
    --num-workers 4 \
    --fp16
