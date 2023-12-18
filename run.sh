#! /bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
start_stage=2
stop_stage=2
step=
input=
output=

echo $PYTHONPATH

echo "$0 $@" # Print the command line for logging

. ./utils/parse_options.sh

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data Preparation"
    python3 preprocess/data_prep.py
fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Extraction"
    python preprocess/audio_preprocess.py --data-config configs/data.yaml \
                                          --spec \
                                          --mel \
                                          --f0 \
                                          --energy \
                                          --stat
fi

if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Training"
    export CUDA_VISIBLE_DEVICES=0,1
    python train_gan.py --data-config configs/svs/data.yaml \
                        --model-config configs/svs/model.yaml \
                        --train-config configs/svs/train.yaml \
                        --num-gpus 2 \
                        --dist-url 'tcp://localhost:30305'
fi

if [ ${start_stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Synthesizing"
    python synthesize.py --exp-name ${name} \
                         --step ${step} \
                         --input ${input} \
                         --output ${output}
fi
