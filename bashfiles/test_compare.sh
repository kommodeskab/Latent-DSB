#!/bin/sh

# SET JOB NAME
#BSUB -J comparison

# select gpu, choose gpuv100 or gpua100 (best)
#BSUB -q gpuv100

# number of GPUs to use
#BSUB -gpu "num=1:mode=exclusive_process"

# number of cores to use
#BSUB -n 12

# gb memory per core
#BSUB -R "rusage[mem=1GB]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 4:00
#BSUB -o hpc/compare.out 
#BSUB -e hpc/compare.err 

module load python3/3.11.9
source .venv/bin/activate

NUM_SAMPLES=256
NUM_STEPS=30
BATCH_SIZE=16

python inference.py --experiment_id=baseline_290525152024   --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS                      --what_test=declip --folder_name=comparison/baseline
python inference.py --experiment_id=GFB_290525152024        --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS                      --what_test=declip --folder_name=comparison/gfb
python inference.py --experiment_id=290525152024            --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=1.0   --what_test=declip --folder_name=comparison/stft_sto
python inference.py --experiment_id=290525152024            --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=0.0   --what_test=declip --folder_name=comparison/stft_det
python inference.py --experiment_id=020625101550            --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=1.0   --what_test=declip --folder_name=comparison/mel_sto
python inference.py --experiment_id=020625101550            --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=0.0   --what_test=declip --folder_name=comparison/mel_det

chgrp -R s214630bjjemiri /work3/s214630/Latent-DSB/
chmod -R 770 /work3/s214630/Latent-DSB/