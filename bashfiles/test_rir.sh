#!/bin/sh

# SET JOB NAME
#BSUB -J test_rir

# select gpu, choose gpuv100 or gpua100 (best)
#BSUB -q gpuv100

# number of GPUs to use
#BSUB -gpu "num=1:mode=exclusive_process"

# number of cores to use
#BSUB -n 12

# gb memory per core
#BSUB -R "rusage[mem=2GB]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 6:00
#BSUB -o hpc/test_rir.out 
#BSUB -e hpc/test_rir.err 

module load python3/3.11.9
source .venv/bin/activate

NUM_SAMPLES=256
BATCH_SIZE=16

python inference.py --experiment_id=baseline --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=1 --what_test=rir --folder_name=test_rir/baseline
# 
# for NUM_STEPS in 1 2 5 10 15 30 50
# do
#   python inference.py --experiment_id=GFB          --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS                    --folder_name=test_rir/gfb_$NUM_STEPS       --what_test=rir
#   python inference.py --experiment_id=130625004303 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=1.0 --folder_name=test_rir/stft_sto_$NUM_STEPS  --what_test=rir
#   python inference.py --experiment_id=130625004256 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=1.0 --folder_name=test_rir/mel_sto_$NUM_STEPS   --what_test=rir
# done

chgrp -R s214630bjjemiri /work3/s214630/Latent-DSB/
chmod -R 770 /work3/s214630/Latent-DSB/