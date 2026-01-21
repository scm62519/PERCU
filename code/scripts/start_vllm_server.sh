#!/bin/bash

# cd scripts 
# ./start_vllm_server.sh 


echo "Activating Conda environment 'vllm-agent'..."

source ~/anaconda3/etc/profile.d/conda.sh 
conda activate vllm-agent


if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'vllm-agent'."
    echo "Please check if the environment exists and anaconda is in your PATH."
    exit 1
fi


export CUDA_VISIBLE_DEVICES=0
echo "Using GPU: $CUDA_VISIBLE_DEVICES"



# OS-Atlas-Pro-7B  GUI-Owl-7B    ShowUI-2B  UI-TARS-1.5-7B        Qwen2.5-VL-7B-Instruct 

MODEL_PATH="/data/zkyao/cmshi/model/GUI-Owl-7B"
echo "Loading model from: $MODEL_PATH"


echo "Starting vLLM server on port 8030..."

vllm serve "$MODEL_PATH" \
    --tensor-parallel-size 1 \
    --port 8030 \
    --host 0.0.0.0 \
    --trust-remote-code


echo "vLLM server stopped."