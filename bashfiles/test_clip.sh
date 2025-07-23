#!/bin/sh

# SET JOB NAME
#BSUB -J test_clip

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
#BSUB -W 2:00
#BSUB -o hpc/test_clip.out 
#BSUB -e hpc/test_clip.err 

module load python3/3.11.9
source .venv/bin/activate

NUM_SAMPLES=256
BATCH_SIZE=16

python inference.py --experiment_id=baseline --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --folder_name=test_clip/baseline --what_test=clip

for NUM_STEPS in 10
do
  python inference.py --experiment_id=GFB          --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS                    --folder_name=test_clip/gfb_$NUM_STEPS       --what_test=clip
  # python inference.py --experiment_id=290525152024 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=0.0 --folder_name=test_clip/stft_det_$NUM_STEPS  --what_test=clip
  # python inference.py --experiment_id=290525152024 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=1.0 --folder_name=test_clip/stft_sto_$NUM_STEPS  --what_test=clip
  # python inference.py --experiment_id=020625101550 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=0.0 --folder_name=test_clip/mel_det_$NUM_STEPS   --what_test=clip 
  # python inference.py --experiment_id=020625101550 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=1.0 --folder_name=test_clip/mel_sto_$NUM_STEPS   --what_test=clip 
done

chgrp -R s214630bjjemiri /work3/s214630/Latent-DSB/
chmod -R 770 /work3/s214630/Latent-DSB/