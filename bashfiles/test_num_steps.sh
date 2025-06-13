#!/bin/sh

# SET JOB NAME
#BSUB -J num_steps

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
#BSUB -W 4:00
#BSUB -o hpc/num_steps.out 
#BSUB -e hpc/num_steps.err 

module load python3/3.11.9
source .venv/bin/activate

LENGTH_SECONDS=4.47
NUM_SAMPLES=256
BATCH_SIZE=16
USE_CLIPPED_LIBRI=True

for NUM_STEPS in 50 30 15 10 5 2 1
do
  python inference.py --experiment_id=GFB_290525152024  --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS                    --folder_name=num_steps/gfb_$NUM_STEPS --what_test=declip
  python inference.py --experiment_id=290525152024      --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=0.0 --folder_name=num_steps/stft_det_$NUM_STEPS --what_test=declip
  python inference.py --experiment_id=290525152024      --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=1.0 --folder_name=num_steps/stft_sto_$NUM_STEPS --what_test=declip
  python inference.py --experiment_id=020625101550      --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=0.0 --folder_name=num_steps/mel_det_$NUM_STEPS --what_test=declip 
  python inference.py --experiment_id=020625101550      --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=$NUM_STEPS --noise_factor=1.0 --folder_name=num_steps/mel_sto_$NUM_STEPS --what_test=declip 

done

chgrp -R s214630bjjemiri /work3/s214630/Latent-DSB/
chmod -R 770 /work3/s214630/Latent-DSB/