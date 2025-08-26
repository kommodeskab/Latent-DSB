#!/bin/sh

# SET JOB NAME
#BSUB -J noise_factor

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
#BSUB -o hpc/noise_factor.out 
#BSUB -e hpc/noise_factor.err 

module load python3/3.11.9
source .venv/bin/activate

NUM_SAMPLES=256
BATCH_SIZE=16

for NOISE_FACTOR in 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
do
  python inference.py --experiment_id=ESDSB_130825172731 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --noise_factor=$NOISE_FACTOR --num_steps=30 --folder_name=test_noise_factor/esdsb_sto_$NOISE_FACTOR --what_test=noise 
done

chgrp -R s214630bjjemiri /work3/s214630/Latent-DSB/
chmod -R 770 /work3/s214630/Latent-DSB/